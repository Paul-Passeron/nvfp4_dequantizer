const std = @import("std");
const st = @import("st");
const builtin = @import("builtin");

// Found the original NVIDIA paper at: <https://arxiv.org/pdf/2509.25149>

// From <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html>:
// The representation of an NVFP4 tensor element x is given by:
// `x = x_e2m1 * s_block * s_global`
// where
//
// `x_e2m1` is the 4-bit value,
// `s_block` is a local FP8 E4M3 scaling factor shared by a block of 16 consecutive elements,
// `s_global` is a global FP32 scaling factor applied to the entire tensor.

// Representations:
// - NVFP4:
//      1 sign bit
//      2 exponent bits
//      1 mantissa bit
//
// - FP8 E4M3:
//      1 sign bit
//      4 exponent bits
//      3 mantissa bits

fn E2M1_from_parts(
    sign: u16,
    e: u16,
    m: u16,
) u16 {
    var bits: u16 = sign;
    if (e == 0) {
        if (m == 1) bits |= 14 << 10;
    } else {
        bits |= ((e + 14) << 10) | (m << 9);
    }
    return bits;
}

fn decodeE2M1_impl(x: u4) f32 {
    const sign: u16 = @as(u16, x >> 3) << 15;
    const e: u16 = (x >> 1) & 0b11;
    const m: u16 = x & 0b1;
    const bits = E2M1_from_parts(sign, e, m);
    return @as(f32, @as(f16, @bitCast(bits)));
}

fn create_decodeE2M1_table() [16]f32 {
    var table: [16]f32 = undefined;
    for (&table, 0..) |*v, i| {
        v.* = decodeE2M1_impl(@as(u4, i));
    }
    return table;
}

const decodeE2M1_table: [16]f32 = create_decodeE2M1_table();

pub fn decodeE2M1(nibble: u4) f32 {
    return decodeE2M1_table[nibble];
}

test "decodeE2M1 covers all 16 values" {
    const expected = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    for (expected, 0..) |e, i| try std.testing.expectEqual(e, decodeE2M1(@intCast(i)));
}

fn E4M3_from_parts(sign: u16, e: u16, m: u16) u16 {
    var bits: u16 = sign;
    if (e == 0b1111 and m == 0b111) {
        bits |= 0x7E00; // NaN
    } else if (e == 0) {
        if (m != 0) {
            const p: u4 = @intCast(15 - @clz(m));
            const frac = m & ((@as(u16, 1) << p) - 1);
            bits |= ((@as(u16, p) + 6) << 10) | (frac << (10 - p));
        }
    } else {
        bits |= ((e + 8) << 10) | (m << 7);
    }
    return bits;
}

pub fn decodeE4M3(byte: u8) f32 {
    const sign: u16 = @as(u16, byte >> 7) << 15;
    const e: u16 = @as(u16, byte >> 3) & 0b1111;
    const m: u16 = byte & 0b111;
    const bits = E4M3_from_parts(sign, e, m);
    return @as(f32, @as(f16, @bitCast(bits)));
}

pub fn unpack(byte: u8) [2]u4 {
    const lo: u4 = @intCast(byte & 0b1111);
    const hi: u4 = @intCast(byte >> 4 & 0b1111);
    return .{ lo, hi };
}

pub fn dequantBlock(
    packed_bytes: *const [8]u8,
    scale: u8,
    inv_global: f32,
    out: *[16]f32,
) void {
    const scale_f = decodeE4M3(scale) * inv_global;
    const V = @Vector(16, f32);
    const v: V = @splat(scale_f);
    for (0.., packed_bytes) |i, byte| {
        const lo, const hi = unpack(byte);
        const lo_f = decodeE2M1(lo);
        const hi_f = decodeE2M1(hi);
        out[2 * i + 0] = lo_f;
        out[2 * i + 1] = hi_f;
    }
    const v2: V = out.*;
    out.* = v2 * v;
}

/// View type over a `File` for a specific NVFP4 encoded layer
pub const NVFP4Tensor = struct {
    weight_global_scale: *const st.TensorInfo,
    weight_scale: *const st.TensorInfo,
    weight_packed: *const st.TensorInfo,

    file: *const st.File,

    pub const BlockReader = struct {
        global_scale: f32,
        block_scale_reader: std.Io.File.Reader,
        packed_reader: std.Io.File.Reader,

        block_scale_buffer: []u8,
        packed_buffer: []u8,
        allocator: std.mem.Allocator,

        layer: *NVFP4Tensor,
        i: usize,

        pub fn deinit(self: *BlockReader) void {
            self.allocator.free(self.block_scale_buffer);
            self.allocator.free(self.packed_buffer);
        }

        pub fn next(self: *BlockReader, out: *[16]f32) !bool {
            if (self.i >= self.layer.weight_scale.byte_len) {
                return false;
            }
            self.i += 1;
            var bytes: [8]u8 = undefined;
            try self.packed_reader.interface.readSliceAll(&bytes);
            const scale = try self.block_scale_reader.interface.takeByte();
            dequantBlock(&bytes, scale, 1.0 / self.global_scale, out);
            return true;
        }
    };

    fn getGlobalScale(self: *NVFP4Tensor) !f32 {
        return self.file.readScalar(f32, self.weight_global_scale);
    }

    pub fn reader(self: *NVFP4Tensor, allocator: std.mem.Allocator) !BlockReader {
        const global_scale = try self.getGlobalScale();

        const block_scale_buffer: []u8 = try allocator.alloc(u8, 1024);
        errdefer allocator.free(block_scale_buffer);

        const packed_buffer: []u8 = try allocator.alloc(u8, 1024);
        errdefer allocator.free(packed_buffer);

        var block_scale_reader = self.file.file.reader(self.file.io, block_scale_buffer);
        try block_scale_reader.seekTo(self.file.headerOffsetBytes() + self.weight_scale.offset);

        var packed_reader = self.file.file.reader(self.file.io, packed_buffer);
        try packed_reader.seekTo(self.file.headerOffsetBytes() + self.weight_packed.offset);

        return .{
            .global_scale = global_scale,
            .block_scale_reader = block_scale_reader,
            .packed_reader = packed_reader,
            .block_scale_buffer = block_scale_buffer,
            .packed_buffer = packed_buffer,
            .allocator = allocator,
            .i = 0,
            .layer = self,
        };
    }

    pub fn from(file: *const st.File, base: []const u8) error{ NotNVFP4, MissingTensorInfo }!?NVFP4Tensor {
        var weight_global_scale: ?*const st.TensorInfo = null;
        var weight_scale: ?*const st.TensorInfo = null;
        var weight_packed: ?*const st.TensorInfo = null;

        var it = file.iterator();
        while (it.next()) |t| {
            if (!std.mem.startsWith(u8, t.name, base)) {
                continue;
            }
            if (std.mem.endsWith(u8, t.name, "weight_global_scale")) {
                weight_global_scale = t;
            }
            if (std.mem.endsWith(u8, t.name, "weight_scale")) {
                weight_scale = t;
            }
            if (std.mem.endsWith(u8, t.name, "weight_packed")) {
                if (t.dtype != .U8) {
                    return error.NotNVFP4;
                }
                weight_packed = t;
            }
        }

        return .{
            .weight_global_scale = weight_global_scale orelse
                return error.MissingTensorInfo,
            .weight_scale = weight_scale orelse
                return error.MissingTensorInfo,
            .weight_packed = weight_packed orelse
                return error.MissingTensorInfo,
            .file = file,
        };
    }
};

fn checkDequantRangeInput(
    packed_bytes: []const u8,
    scales: []const u8,
    out: []f32,
) bool {
    if (packed_bytes.len % 8 != 0) return false;
    if (packed_bytes.len / 8 != scales.len) return false;
    if (out.len != scales.len * 16) return false;
    return true;
}

pub fn dequantRange(
    packed_bytes: []const u8,
    scales: []const u8,
    inv_global: f32,
    out: []f32,
) void {
    if (builtin.mode == .debug) {
        if (!checkDequantRangeInput(packed_bytes, scales, out)) {
            @panic("Invalid dequant range input");
        }
    }
    // Dumb way for the moment, using what we already have
    for (scales, 0..) |scale, i| {
        const bytes_slice = packed_bytes[i * 8 ..][0..8];
        const out_slice = out[i * 16 ..][0..16];
        dequantBlock(bytes_slice, scale, inv_global, out_slice);
    }
}

const BLOCK_COUNT = 16;
const PACKED_IN_BYTES = 2;
const BLOCK_SIZE = BLOCK_COUNT / PACKED_IN_BYTES;

/// Inspired by Std.Io.File.Reader
pub const Reader = struct {
    blocks: NVFP4Tensor.BlockReader,
    interface: std.Io.Reader,

    index_within_block: usize = @sizeOf([16]f32),
    block: [16]f32 = undefined,

    const Self = @This();

    fn stream(
        io_reader: *std.Io.Reader,
        w: *std.Io.Writer,
        limit: std.Io.Limit,
    ) !usize {
        const r: *Self = @alignCast(
            @fieldParentPtr("interface", io_reader),
        );
        return r.read_impl(w, limit);
    }

    fn read_impl(
        self: *Self,
        w: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        const block_bytes: []u8 = std.mem.asBytes(&self.block);
        if (self.index_within_block == block_bytes.len) {
            if (!try self.blocks.next(&self.block))
                return error.EndOfStream;
            self.index_within_block = 0;
        }
        const slice = limit.slice(block_bytes[self.index_within_block..]);
        const n = try w.write(slice);
        self.index_within_block += n;
        return n;
    }

    pub fn initInterface(buffer: []u8) std.Io.Reader {
        return .{
            .vtable = &.{ .stream = stream },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
    }

    pub fn init(blocks: NVFP4Tensor.BlockReader, buffer: []u8) Reader {
        return .{
            .blocks = blocks,
            .interface = initInterface(buffer),
        };
    }
};
