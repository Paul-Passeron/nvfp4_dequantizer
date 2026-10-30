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

fn checkDequantRangeInput(
    packed_bytes: []const u8,
    scales: []const u8,
    out: []f32,
) error{InvalidDequantRanges}!void {
    const err = error.InvalidDequantRanges;
    if (packed_bytes.len % 8 != 0) return err;
    if (packed_bytes.len / 8 != scales.len) return err;
    if (out.len != scales.len * 16) return err;
}

pub fn dequantRange(
    packed_bytes: []const u8,
    scales: []const u8,
    inv_global: f32,
    out: []f32,
) error{InvalidDequantRanges}!void {
    try checkDequantRangeInput(packed_bytes, scales, out);
    // Dumb way for the moment, using what we already have
    for (scales, 0..) |scale, i| {
        const bytes_slice = packed_bytes[i * 8 ..][0..8];
        const out_slice = out[i * 16 ..][0..16];
        dequantBlock(bytes_slice, scale, inv_global, out_slice);
    }
}

pub const Tensor = struct {
    weight_packed: *const st.Tensor,
    weight_scale: *const st.Tensor,
    weight_global_scale: *const st.Tensor,

    fn get_tensor_from_registry(
        a: std.mem.Allocator,
        registry: *const st.Registry,
        base: []const u8,
        comptime end: []const u8,
    ) error{MissingTensorInfo}!*const st.Tensor {
        const weight_packed_name =
            std.fmt.allocPrint(a, "{s}." ++ end, .{base}) catch unreachable;
        const weight_packed = registry.get(weight_packed_name) orelse
            return error.MissingTensorInfo;
        return weight_packed;
    }

    /// returns null when couldn't find `<base>.weight_packed` tensor
    pub fn fromRegistry(
        registry: *const st.Registry,
        base: []const u8,
    ) error{MissingTensorInfo}!?Tensor {
        var scratch_b: [512]u8 = undefined;
        var scratch: std.heap.FixedBufferAllocator = .init(&scratch_b);
        const a = scratch.allocator();

        const weight_packed =
            get_tensor_from_registry(
                a,
                registry,
                base,
                "weight_packed",
            ) catch return null;

        scratch.reset();
        const weight_scale =
            try get_tensor_from_registry(a, registry, base, "weight_scale");

        scratch.reset();
        const weight_global_scale =
            try get_tensor_from_registry(a, registry, base, "weight_global_scale");

        return .{
            .weight_packed = weight_packed,
            .weight_scale = weight_scale,
            .weight_global_scale = weight_global_scale,
        };
    }

    pub fn shape(self: *const Tensor) [2]u64 {
        const n = self.weight_packed.shape[0];
        const k = self.weight_packed.shape[1] * 2;
        return .{ n, k };
    }

    pub fn dequantByteLen(self: *const Tensor) u64 {
        // 2 samples per u8 (u4 + u4)
        // and then 32 bits per f32 sample (=4 bytes)
        // so packed byte len * 8
        return self.weight_packed * 8;
    }

    pub fn reader(
        self: *const Tensor,
        io: std.Io,
        buffers: st.TensorReader.Buffers,
    ) !TensorReader {
        return TensorReader.init(io, self, buffers);
    }
};

test "can get tensor from registry" {
    const a = std.testing.allocator;
    const filename = "nvfp4/model.safetensors";
    const base = "model.layers.0.self_attn.q_proj";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();
    _ = try Tensor.fromRegistry(&reg, base);
}

const BLOCK_COUNT = 16;
const DEQUANT_BLOCK_BYTE_LEN = @sizeOf(f32) * BLOCK_COUNT;
const PACKED_PER_BLOCK = BLOCK_COUNT / 2; // 2 u4 s packed in a byte

pub const TensorReader = struct {
    packed_reader: st.TensorReader,
    scale_reader: st.TensorReader,
    inv_global: f32,
    remaining_blocks: u64,
    pending: [BLOCK_COUNT]f32 = undefined,
    pending_index: usize = @sizeOf([BLOCK_COUNT]f32),

    interface: std.Io.Reader,

    pub const Buffers = struct {
        packed_bytes: []u8,
        scales: []u8,
        interface: []u8 = &.{},
    };

    fn stream(
        io_reader: *std.Io.Reader,
        io_writer: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        var this: *TensorReader = @fieldParentPtr("interface", io_reader);
        return this.streamImpl(io_writer, limit);
    }

    fn streamImpl(
        self: *TensorReader,
        io_writer: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {

        // Handle last block if it still hasn't been completely read
        const pending_bytes: []u8 = std.mem.asBytes(&self.pending);
        if (self.pending_index < pending_bytes.len) {
            const n = try io_writer.write(limit.slice(pending_bytes[self.pending_index..]));
            self.pending_index += n;
            return n;
        }

        // No pending block so if no remaining blocks then we truly are at the end
        if (self.remaining_blocks == 0) {
            return error.EndOfStream;
        }

        const dest = limit.slice(try io_writer.writableSliceGreedy(1));
        const blocks_to_write = dest.len / DEQUANT_BLOCK_BYTE_LEN;

        if (blocks_to_write == 0) {
            const packed_bytes: [PACKED_PER_BLOCK]u8 = undefined;
            _ = packed_bytes;
            @panic("TODO: read a single block inside pending block");
        }

        @panic("TODO: Handle the general case, i.e. reading at least one new block");
    }

    fn readInputs(self: *TensorReader, packed_bytes: []u8, scales: []u8) !void {
        _ = self;
        _ = packed_bytes;
        _ = scales;
        @panic("TODO: Read as much as possible inside packed_bytes and scales");
    }

    fn initInterface(buffer: []u8) std.Io.Reader {
        return .{
            .buffer = buffer,
            .end = 0,
            .seek = 0,
            .vtable = .{
                .stream = stream,
            },
        };
    }

    fn deinit(self: *TensorReader) void {
        self.packed_reader.deinit();
        self.scale_reader.deinit();
    }

    fn init(
        io: std.Io,
        tensor: *const Tensor,
        buffers: Buffers,
    ) !TensorReader {
        const packed_reader: st.TensorReader = .init(
            io,
            tensor.weight_packed,
            .{ .file = buffers.packed_bytes },
        );
        const scale_reader: st.TensorReader = .init(
            io,
            tensor.weight_scale,
            .{ .file = buffers.scales },
        );
        const inv_global = try tensor.weight_global_scale.readScalar(io, f32);

        const remaining_blocks = tensor.weight_scale.byte_len;
        return .{
            .packed_reader = packed_reader,
            .scale_reader = scale_reader,
            .inv_global = inv_global,
            .remaining_blocks = remaining_blocks,
            .interface = initInterface(buffers.interface),
        };
    }
};
