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
    // const v: u32 = x;
    //     const s = (v >> 3) & 1;
    //     const e1 = (v >> 2) & 1;
    //     const e0 = (v >> 1) & 1;
    //     const m = v & 1;
    //
    //     const t = ~e1 & (e0 | m) & 1;
    //
    //     var tt = t | (t << 1);
    //     tt |= tt << 2;
    //     tt |= tt << 4;
    //
    //     const exp = (e1 << 7) | (tt & 0b0111_1110) | e0;
    //     const bits = (s << 31) | (exp << 23) | ((m & (e0 | e1)) << 22);
    //     return @bitCast(bits);
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

fn split_bytes_into_nibbles(
    comptime n: comptime_int,
    comptime T: type,
    bytes: @Vector(n, u8),
) @Vector(n * 2, T) {
    const mask: @Vector(8, u8) = @splat(0xF);
    const hi = bytes & mask;
    const lo = (bytes >> @splat(4)) & mask;
    return std.simd.interlace(.{ lo, hi });
}

fn e2m1Range(
    comptime n: comptime_int,
    packed_bytes: *const [n * 8]u8,
) @Vector(n * 16, f32) {
    const V = @Vector(n * 16, u32);
    const bytes: @Vector(n * 8, u8) = packed_bytes.*;
    const v: V = split_bytes_into_nibbles(n * 8, u32, bytes);
    const s: V = (v >> @splat(3)) & @as(V, @splat(1));
    const e1 = (v >> @splat(2)) & @as(V, @splat(1));
    const e0 = (v >> @splat(1)) & @as(V, @splat(1));
    const m = v & @as(V, @splat(1));
    const t = ~e1 & (e0 | m) & @as(V, @splat(1));
    var tt = t | (t << @as(V, @splat(1)));
    tt |= tt << @splat(2);
    tt |= tt << @splat(4);
    const exp = (e1 << @splat(7)) | (tt & @as(V, @splat(0b0111_1110))) | e0;
    const bits = (s << @splat(31)) | (exp << @splat(23)) | ((m & (e0 | e1)) << @splat(22));
    return @bitCast(bits);
}

test "e2m1" {
    const values: [8]u8 = .{ 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90 };
    const split: [16]u4 = .{ 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0 };
    const res = e2m1Range(1, &values);
    const simd: [16]f32 = res;
    for (simd, split) |a, b| {
        const decode = decodeE2M1(b);
        try std.testing.expectEqual(a, decode);
    }
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
        return self.weight_packed.byte_len * 8;
    }

    pub fn reader(
        self: *const Tensor,
        io: std.Io,
        buffers: TensorReader.Buffers,
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
const CHUNK_BLOCKS = 256;

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
            var packed_bytes: [PACKED_PER_BLOCK]u8 = undefined;
            var scale: [1]u8 = undefined;
            try self.readInputs(&packed_bytes, &scale);
            dequantBlock(&packed_bytes, scale[0], self.inv_global, &self.pending);
            self.remaining_blocks -= 1;
            const n = try io_writer.write(limit.slice(pending_bytes));
            self.pending_index = n;
            return n;
        }

        const k: usize = @intCast(@min(
            blocks_to_write,
            CHUNK_BLOCKS,
            self.remaining_blocks,
        ));

        var packed_bytes: [CHUNK_BLOCKS * PACKED_PER_BLOCK]u8 = undefined;
        var scales: [CHUNK_BLOCKS]u8 = undefined;
        var out: [CHUNK_BLOCKS * BLOCK_COUNT]f32 = undefined;
        try self.readInputs(packed_bytes[0 .. k * PACKED_PER_BLOCK], scales[0..k]);
        dequantRange(
            packed_bytes[0 .. k * PACKED_PER_BLOCK],
            scales[0..k],
            self.inv_global,
            out[0 .. k * BLOCK_COUNT],
        ) catch unreachable;

        const n = k * DEQUANT_BLOCK_BYTE_LEN;
        // we know that n fits because
        // n = k * DEQUANT_BLOCK_BYTE_LEN and k <= blocks_to_write
        // and blocks_to_write = dest.len / DEQUANT_BLOCK_BYTE_LEN
        // so n <= dest.len
        @memcpy(dest[0..n], std.mem.sliceAsBytes(out[0 .. k * BLOCK_COUNT]));
        io_writer.advance(n);
        self.remaining_blocks -= k;
        return n;
    }

    fn readInputs(self: *TensorReader, packed_bytes: []u8, scales: []u8) !void {
        self.packed_reader.interface.readSliceAll(packed_bytes) catch |e|
            return mapErr(e);
        self.scale_reader.interface.readSliceAll(scales) catch |e|
            return mapErr(e);
    }

    fn mapErr(e: std.Io.Reader.Error) std.Io.Reader.Error {
        return switch (e) {
            error.EndOfStream => error.ReadFailed,
            else => e,
        };
    }

    fn initInterface(buffer: []u8) std.Io.Reader {
        return .{
            .buffer = buffer,
            .end = 0,
            .seek = 0,
            .vtable = &.{ .stream = stream },
        };
    }

    pub fn deinit(self: *TensorReader) void {
        self.packed_reader.deinit();
        self.scale_reader.deinit();
    }

    fn init(
        io: std.Io,
        tensor: *const Tensor,
        buffers: Buffers,
    ) !TensorReader {
        const packed_reader: st.TensorReader = try .init(
            io,
            tensor.weight_packed,
            .{ .file = buffers.packed_bytes },
        );
        const scale_reader: st.TensorReader = try .init(
            io,
            tensor.weight_scale,
            .{ .file = buffers.scales },
        );
        const inv_global = 1.0 / try tensor.weight_global_scale.readScalar(io, f32);

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

test "nvfp4 and f32 are the same" {
    const a = std.testing.allocator;
    const base = "model.layers.0.self_attn.q_proj";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var nvfp4_reg: st.Registry = try .fromPath(a, io, "nvfp4/model.safetensors");
    defer nvfp4_reg.deinit();
    var f32_reg: st.Registry = try .fromPath(
        a,
        io,
        "converted_f32/model-00001-of-00001.safetensors",
    );
    defer f32_reg.deinit();

    const t = try Tensor.fromRegistry(&nvfp4_reg, base) orelse
        return error.MissingTensor;
    const expected = f32_reg.get(base ++ ".weight") orelse
        return error.MissingTensor;

    try std.testing.expectEqual(st.DataType.F32, expected.dtype);
    try std.testing.expectEqualSlices(u64, expected.shape, &t.shape());
    try std.testing.expectEqual(expected.byte_len, t.dequantByteLen());

    var packed_buffer: [4096]u8 = undefined;
    var scales_buffer: [4096]u8 = undefined;
    var actual_reader = try t.reader(io, .{
        .packed_bytes = &packed_buffer,
        .scales = &scales_buffer,
    });
    defer actual_reader.deinit();

    var expected_buffer: [4096]u8 = undefined;
    var expected_reader: st.TensorReader = try .init(io, expected, .{
        .file = &expected_buffer,
    });
    defer expected_reader.deinit();

    var actual_chunk: [64 * 1024]u8 align(@alignOf(f32)) = undefined;
    var expected_chunk: [64 * 1024]u8 align(@alignOf(f32)) = undefined;

    var left = expected.byte_len;
    while (left > 0) {
        const n: usize = @intCast(@min(left, actual_chunk.len));
        try actual_reader.interface.readSliceAll(actual_chunk[0..n]);
        try expected_reader.interface.readSliceAll(expected_chunk[0..n]);

        const actual_values: []f32 = std.mem.bytesAsSlice(f32, actual_chunk[0..n]);
        const expected_values: []f32 = std.mem.bytesAsSlice(f32, expected_chunk[0..n]);
        for (expected_values, actual_values) |e, v| {
            try std.testing.expectApproxEqRel(e, v, 1e-3);
        }
        left -= n;
    }

    try std.testing.expectEqual(
        @as(usize, 0),
        try actual_reader.interface.discardRemaining(),
    );
}

test "nvfp4 reader through a small interface buffer" {
    const a = std.testing.allocator;
    const base = "model.layers.0.self_attn.q_proj";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var nvfp4_reg: st.Registry = try .fromPath(a, io, "nvfp4/model.safetensors");
    defer nvfp4_reg.deinit();
    var f32_reg: st.Registry = try .fromPath(
        a,
        io,
        "converted_f32/model-00001-of-00001.safetensors",
    );
    defer f32_reg.deinit();

    const t = try Tensor.fromRegistry(&nvfp4_reg, base) orelse
        return error.MissingTensor;
    const expected = f32_reg.get(base ++ ".weight") orelse
        return error.MissingTensor;

    var packed_buffer: [4096]u8 = undefined;
    var scales_buffer: [4096]u8 = undefined;
    var interface_buffer: [8]u8 = undefined;
    var actual_reader = try t.reader(io, .{
        .packed_bytes = &packed_buffer,
        .scales = &scales_buffer,
        .interface = &interface_buffer,
    });
    defer actual_reader.deinit();

    var expected_file_buffer: [4096]u8 = undefined;
    var expected_interface_buffer: [4096]u8 = undefined;
    var expected_reader: st.TensorReader = try .init(io, expected, .{
        .file = &expected_file_buffer,
        .interface = &expected_interface_buffer,
    });
    defer expected_reader.deinit();

    const value_count = 1000;
    for (0..value_count) |_| {
        const e: f32 = @bitCast(try expected_reader.interface.takeInt(u32, .little));
        const v: f32 = @bitCast(try actual_reader.interface.takeInt(u32, .little));
        try std.testing.expectApproxEqRel(e, v, 1e-3);
    }

    var actual_chunk: [4096]u8 align(@alignOf(f32)) = undefined;
    var expected_chunk: [4096]u8 align(@alignOf(f32)) = undefined;
    try actual_reader.interface.readSliceAll(&actual_chunk);
    try expected_reader.interface.readSliceAll(&expected_chunk);

    const actual_values: []f32 = std.mem.bytesAsSlice(f32, &actual_chunk);
    const expected_values: []f32 = std.mem.bytesAsSlice(f32, &expected_chunk);
    for (expected_values, actual_values) |e, v| {
        try std.testing.expectApproxEqRel(e, v, 1e-3);
    }
}
