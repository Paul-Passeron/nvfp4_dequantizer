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

pub inline fn decodeE2M1(nibble: u4) f32 {
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
    out: *align(1) [16]f32,
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
    out: []align(1) f32,
) error{InvalidDequantRanges}!void {
    const err = error.InvalidDequantRanges;
    if (packed_bytes.len % 8 != 0) return err;
    if (packed_bytes.len / 8 != scales.len) return err;
    if (out.len != scales.len * 16) return err;
}

fn splitBytesIntoNibbles(
    comptime n: comptime_int,
    bytes: @Vector(n, u8),
) @Vector(n * 2, u8) {
    const mask: @Vector(n, u8) = @splat(0xF);
    const lo = bytes & mask;
    const hi = (bytes >> @splat(4));
    const interleaved: @Vector(n * 2, u8) =
        @shuffle(u8, lo, hi, interleaveMask(n));
    return interleaved;
}

pub const e2m1Alg = enum { look_up_table, bits, tbl };

fn e2m1ByteTable(comptime shift: u5) @Vector(16, u8) {
    var t: [16]u8 = undefined;
    for (&t, decodeE2M1_table) |*x, f| {
        const bits: u32 = @bitCast(f);
        std.debug.assert(bits & 0xFFFF == 0);
        x.* = @truncate(bits >> shift);
    }
    return t;
}

const e2m1_byte3_table = e2m1ByteTable(24);
const e2m1_byte2_table = e2m1ByteTable(16);

inline fn tblAsmImpl(table: @Vector(16, u8), idx: @Vector(16, u8)) @Vector(16, u8) {
    return asm ("tbl %[out].16b, {%[t].16b}, %[i].16b"
        : [out] "=w" (-> @Vector(16, u8)),
        : [t] "w" (table),
          [i] "w" (idx),
    );
}

inline fn tblGeneralImpl(table: @Vector(16, u8), idx: @Vector(16, u8)) @Vector(16, u8) {
    var res: [16]u8 = undefined;
    inline for (0..16) |i| res[i] = table[idx[i]];
    return res;
}

// From <https://ziggit.dev/t/simd-and-lookup-tables/15852/4>
inline fn tbl(table: @Vector(16, u8), idx: @Vector(16, u8)) @Vector(16, u8) {
    if (builtin.cpu.arch == .aarch64) {
        return tblAsmImpl(table, idx);
    } else {
        return tblGeneralImpl(table, idx);
    }
}

fn interleaveMask(comptime n: comptime_int) [n * 2]i32 {
    var idx: [n * 2]i32 = undefined;
    for (0..n) |i| {
        idx[2 * i + 0] = @intCast(i);
        idx[2 * i + 1] = ~@as(i32, @intCast(i));
    }
    return idx;
}

inline fn merge2U8aAsU16(
    comptime n: comptime_int,
    lo: @Vector(n, u8),
    hi: @Vector(n, u8),
) @Vector(n, u16) {
    return @bitCast(@shuffle(u8, lo, hi, interleaveMask(n)));
}

inline fn decodeE2M1FromIndices(idx: @Vector(16, u8)) @Vector(16, u32) {
    const b3 = tbl(e2m1_byte3_table, idx);
    const b2 = tbl(e2m1_byte2_table, idx);
    const hi16 = merge2U8aAsU16(16, b2, b3);
    return (@as(@Vector(16, u32), hi16) << @splat(16));
}

fn e2m1Range(
    comptime n: comptime_int,
    algo: e2m1Alg,
    packed_bytes: *const [n]u8,
) @Vector(n * 2, f32) {
    switch (algo) {
        .bits => {
            const V = @Vector(n * 2, u8);
            const W = @Vector(n * 2, u32);
            const bytes: @Vector(n, u8) = packed_bytes.*;
            const v: V = splitBytesIntoNibbles(n, bytes);
            const s: V = (v >> @splat(3)) & @as(V, @splat(1));
            const e = (v >> @splat(1)) & @as(V, @splat(3));
            const m = v & @as(V, @splat(1));
            const zero: V = @splat(0);
            const is_sub = e == zero;
            const exp = @select(u8, is_sub, m * @as(V, @splat(126)), e + @as(V, @splat(126)));
            const mant = @select(u8, is_sub, zero, m);
            const byte3 = (s << @splat(7)) | (exp >> @splat(1));
            const byte2 = ((exp & @as(V, @splat(1))) << @splat(7)) | (mant << @splat(6));
            const bits = (@as(W, byte3) << @splat(24)) | (@as(W, byte2) << @splat(16));
            return @bitCast(bits);
        },
        .look_up_table => {
            var res: [n * 2]f32 = undefined;
            inline for (packed_bytes, 0..) |b, i| {
                const lo: u4 = @intCast(b & 0b1111);
                const hi: u4 = @intCast(b >> 4);
                res[2 * i + 0] = decodeE2M1_table[lo];
                res[2 * i + 1] = decodeE2M1_table[hi];
            }
            return res;
        },
        .tbl => {
            comptime std.debug.assert(n * 2 % 16 == 0);
            const bytes: @Vector(n, u8) = packed_bytes.*;
            const nibbles: [n * 2]u8 = splitBytesIntoNibbles(n, bytes);
            var res: [n * 2]u32 = undefined;
            inline for (0..n * 2 / 16) |k| {
                const idx: @Vector(16, u8) = nibbles[k * 16 ..][0..16].*;
                res[k * 16 ..][0..16].* = decodeE2M1FromIndices(idx);
            }
            const bits: @Vector(n * 2, u32) = res;
            return @bitCast(bits);
        },
    }
}

test "e2m1 tbl" {
    var values: [128]u8 = undefined;
    // even bytes then odd bytes: covers every lo and hi nibble
    for (&values, 0..) |*v, i| v.* = @intCast(i * 2 + (i >> 6));
    const res: [256]f32 = e2m1Range(128, .tbl, &values);
    const expected: [256]f32 = e2m1Range(128, .look_up_table, &values);
    for (res, expected) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    }
}

test "e2m1 lut" {
    const values: [8]u8 = .{ 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90 };
    var split: [16]u4 = undefined;
    for (values, 0..) |v, i| {
        const lo, const hi = unpack(v);
        split[2 * i + 0] = lo;
        split[2 * i + 1] = hi;
    }

    const res = e2m1Range(8, .look_up_table, &values);
    const simd: [16]f32 = res;
    for (simd, split) |a, b| {
        const decode = decodeE2M1(b);
        try std.testing.expectEqual(a, decode);
    }
}

test "e2m1 bits" {
    const values: [8]u8 = .{ 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x90 };
    var split: [16]u4 = undefined;
    for (values, 0..) |v, i| {
        const lo, const hi = unpack(v);
        split[2 * i + 0] = lo;
        split[2 * i + 1] = hi;
    }

    const res = e2m1Range(8, .bits, &values);
    const simd: [16]f32 = res;
    for (simd, split) |a, b| {
        const decode = decodeE2M1(b);
        try std.testing.expectEqual(a, decode);
    }
}

fn splat(comptime t: type, val: anytype) t {
    return @as(t, @splat(val));
}

fn e4m3RangeImpl(
    comptime n: comptime_int,
    bytes: *const [n]u8,
) @Vector(n, f32) {
    const V = @Vector(n, u32);
    const B = @Vector(n, bool);
    const S = @Vector(n, u5);
    const x: @Vector(n, u32) = bytes.*;
    const sign: V = (x & splat(V, 0x80)) << @splat(24);
    const e: V = (x >> @splat(3)) & splat(V, 0xF);
    const m: V = x & splat(V, 0x7);
    const normal: V = ((e + splat(V, 120)) << splat(S, 23)) |
        (m << splat(S, 20));
    const p: S = @intCast(splat(V, 31) - @clz(m | splat(V, 1)));
    const subnormal: V = ((splat(V, 118) + @as(V, p)) << splat(S, 23)) |
        (((m << (splat(S, 3) - p)) & splat(V, 0x7)) << splat(S, 20));
    const subnormal_or_zero: V = @select(u32, m == splat(V, 0), splat(V, 0), subnormal);
    const magnitude: V = @select(u32, e == splat(V, 0), subnormal_or_zero, normal);
    const is_nan: B = (e == splat(V, 0xF)) & (m == splat(V, 0x7));
    const or_mask = @select(u32, is_nan, splat(V, 0x7FC0_0000), magnitude);
    const res: V = sign | or_mask;
    return @bitCast(res);
}

test "decoding e4m3 works" {
    for (0..256) |v| {
        const val: u8 = @intCast(v);
        const simd: [1]f32 = e4m3RangeImpl(1, &.{val});
        const expected: u32 = @bitCast(decodeE4M3(val));

        try std.testing.expectEqual(@as(u32, @bitCast(simd[0])), expected);
    }
}

fn shuffleMask(comptime n: comptime_int, comptime m: comptime_int) [n * m]i32 {
    var indices: [n * m]i32 = undefined;
    for (&indices, 0..) |*index, i| index.* = @intCast(i / m);
    return indices;
}

fn e4m3Range(
    comptime n: comptime_int,
    bytes: *const [n]u8,
    scale: f32,
) @Vector(n, f32) {
    var sg: @Vector(n, f32) = undefined;
    inline for (0..n) |k| {
        sg[k] = e4m3_table[bytes[k]];
    }
    return sg * splat(@TypeOf(sg), scale);
}

fn broadcast(
    comptime n: comptime_int,
    comptime m: comptime_int,
    comptime t: type,
    v: @Vector(n, t),
) @Vector(n * m, t) {
    return @shuffle(t, v, undefined, shuffleMask(n, m));
}

pub const DequantAlgo = enum { sequential, simd };

const e4m3_table: [256]f32 = blk: {
    @setEvalBranchQuota(20_000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*x, i| x.* = decodeE4M3(@intCast(i));
    break :blk t;
};

inline fn dequantBlocks(
    comptime B: comptime_int,
    algo: e2m1Alg,
    packed_bytes: *const [B * 8]u8,
    scales: *const [B]u8,
    inv_global: f32,
    out: *align(1) [B * 16]f32,
) void {
    const dequantized = e2m1Range(B * 8, algo, packed_bytes);
    const scale_vector = e4m3Range(B, scales, inv_global);
    const broadcasted = broadcast(B, 16, f32, scale_vector);
    const res = dequantized * broadcasted;
    out.* = res;
}

const lengths_size: comptime_int = blk: {
    var n = std.simd.suggestVectorLength(f32) orelse 16;
    var pot = 0;
    while (n > 0) {
        pot += 1;
        n /= 2;
    }
    break :blk @intCast(pot);
};

fn compute_vec_lengths() [lengths_size]u32 {
    var res: [lengths_size]u32 = undefined;
    for (0..lengths_size) |i| {
        const p: i32 = lengths_size - i - 1;
        res[i] = @intCast(std.math.powi(i32, 2, p) catch unreachable);
    }
    return res;
}

const vec_lengths: [lengths_size]u32 = compute_vec_lengths();

noinline fn dequantRangeSimd(
    algo: e2m1Alg,
    packed_bytes: []const u8,
    scales: []const u8,
    inv_global: f32,
    out: []align(1) f32,
) void {
    var remaining_blocks: usize = scales.len;
    inline for (vec_lengths) |C| {
        while (remaining_blocks >= C) {
            const current_block = scales.len - remaining_blocks;
            dequantBlocks(
                C,
                algo,
                packed_bytes[current_block * 8 ..][0 .. C * 8],
                scales[current_block..][0..C],
                inv_global,
                out[current_block * 16 ..][0 .. C * 16],
            );
            remaining_blocks -= C;
        }
        if (remaining_blocks == 0) return;
    }
}

const default_e2m1_algo: e2m1Alg = blk: {
    if (builtin.cpu.arch == .aarch64) {
        break :blk .tbl;
    } else {
        break :blk .look_up_table;
    }
};

pub const DequantConfig = struct {
    e2m1: e2m1Alg = default_e2m1_algo,
    ordering: DequantAlgo,
};

pub fn dequantRange(
    config: DequantConfig,
    packed_bytes: []const u8,
    scales: []const u8,
    inv_global: f32,
    out: []align(1) f32,
) error{InvalidDequantRanges}!void {
    try checkDequantRangeInput(packed_bytes, scales, out);
    switch (config.ordering) {
        .sequential => {
            for (scales, 0..) |scale, i| {
                const bytes_slice = packed_bytes[i * 8 ..][0..8];
                const out_slice = out[i * 16 ..][0..16];
                dequantBlock(bytes_slice, scale, inv_global, out_slice);
            }
        },
        .simd => {
            dequantRangeSimd(config.e2m1, packed_bytes, scales, inv_global, out);
        },
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
    ) error{ MissingTensorInfo, OutOfMemory }!*const st.Tensor {
        const weight_packed_name =
            try a.print("{s}." ++ end, .{base});
        defer a.free(weight_packed_name);
        const weight_packed = registry.get(weight_packed_name) orelse
            return error.MissingTensorInfo;
        return weight_packed;
    }

    /// returns null when couldn't find `<base>.weight_packed` tensor
    pub fn fromRegistry(
        registry: *const st.Registry,
        base: []const u8,
    ) error{ MissingTensorInfo, OutOfMemory }!?Tensor {
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
        config: DequantConfig,
    ) !TensorReader {
        return TensorReader.init(io, self, buffers, config);
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

    config: DequantConfig,

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
            const input = try self.readInputs(PACKED_PER_BLOCK, 1);
            const packed_bytes = input.bytes[0..PACKED_PER_BLOCK];
            const scale = input.scales[0];
            dequantBlock(packed_bytes, scale, self.inv_global, &self.pending);
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

        const n = k * DEQUANT_BLOCK_BYTE_LEN;
        // we know that n fits because
        // n = k * DEQUANT_BLOCK_BYTE_LEN and k <= blocks_to_write
        // and blocks_to_write = dest.len / DEQUANT_BLOCK_BYTE_LEN
        // so n <= dest.len

        const input = try self.readInputs(k * PACKED_PER_BLOCK, k);
        const dest_f32: []align(1) f32 = std.mem.bytesAsSlice(f32, dest[0..n]);
        dequantRange(
            self.config,
            input.bytes,
            input.scales,
            self.inv_global,
            dest_f32,
        ) catch unreachable;

        io_writer.advance(n);
        self.remaining_blocks -= k;
        return n;
    }

    const Input = struct {
        bytes: []const u8,
        scales: []const u8,
    };

    fn readInputs(self: *TensorReader, packed_len: usize, scale_len: usize) !Input {
        const bytes = self.packed_reader.take(packed_len) catch |e| return mapErr(e);
        const scales = self.scale_reader.take(scale_len) catch |e| return mapErr(e);
        return .{ .bytes = bytes, .scales = scales };
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
        config: DequantConfig,
    ) !TensorReader {
        var packed_reader: st.TensorReader = try .init(
            io,
            tensor.weight_packed,
            .{ .file = buffers.packed_bytes },
        );
        errdefer packed_reader.deinit();
        var scale_reader: st.TensorReader = try .init(
            io,
            tensor.weight_scale,
            .{ .file = buffers.scales },
        );
        errdefer scale_reader.deinit();
        const inv_global = 1.0 / try tensor.weight_global_scale.readScalar(io, f32);

        const remaining_blocks = tensor.weight_scale.byte_len;
        return .{
            .packed_reader = packed_reader,
            .scale_reader = scale_reader,
            .inv_global = inv_global,
            .remaining_blocks = remaining_blocks,
            .interface = initInterface(buffers.interface),
            .config = config,
        };
    }
};

test "nvfp4 and f32 are the same sequential" {
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
    }, .{ .ordering = .sequential });
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

test "nvfp4 and f32 are the same simd" {
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
    }, .{ .ordering = .simd });
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

test "nvfp4 is the same for sequential and simd" {
    const a = std.testing.allocator;
    const base = "model.layers.0.self_attn.q_proj";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var nvfp4_reg: st.Registry = try .fromPath(a, io, "nvfp4/model.safetensors");
    defer nvfp4_reg.deinit();

    const t = try Tensor.fromRegistry(&nvfp4_reg, base) orelse
        return error.MissingTensor;

    var packed_buffer: [4096]u8 = undefined;
    var scales_buffer: [4096]u8 = undefined;
    var actual_reader = try t.reader(io, .{
        .packed_bytes = &packed_buffer,
        .scales = &scales_buffer,
    }, .{ .ordering = .simd });
    defer actual_reader.deinit();

    var expected_packed_buffer: [4096]u8 = undefined;
    var expected_scales_buffer: [4096]u8 = undefined;
    var expected_reader = try t.reader(io, .{
        .packed_bytes = &expected_packed_buffer,
        .scales = &expected_scales_buffer,
    }, .{ .ordering = .sequential });
    defer expected_reader.deinit();

    var actual_chunk: [64 * 1024]u8 align(@alignOf(f32)) = undefined;
    var expected_chunk: [64 * 1024]u8 align(@alignOf(f32)) = undefined;

    var left = t.dequantByteLen();
    while (left > 0) {
        const n: usize = @intCast(@min(left, actual_chunk.len));
        try actual_reader.interface.readSliceAll(actual_chunk[0..n]);
        try expected_reader.interface.readSliceAll(expected_chunk[0..n]);

        const actual_values: []f32 = std.mem.bytesAsSlice(f32, actual_chunk[0..n]);
        const expected_values: []f32 = std.mem.bytesAsSlice(f32, expected_chunk[0..n]);
        for (expected_values, actual_values) |e, v| {
            try std.testing.expectApproxEqRel(e, v, 1e-6);
        }
        left -= n;
    }

    try std.testing.expectEqual(
        @as(usize, 0),
        try actual_reader.interface.discardRemaining(),
    );
    try std.testing.expectEqual(
        @as(usize, 0),
        try expected_reader.interface.discardRemaining(),
    );
}

test "nvfp4 reader through a small interface buffer sequential" {
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
    }, .{ .ordering = .sequential });
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

test "nvfp4 reader through a small interface buffer simd" {
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
    }, .{ .ordering = .simd });
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

pub fn getAllNVFP4Tensors(
    a: std.mem.Allocator,
    r: *const st.Registry,
) ![]Tensor {
    var tensors = std.ArrayList(Tensor).empty;
    errdefer tensors.deinit(a);

    var it = r.iterator();
    while (it.next()) |t| {
        if (std.mem.endsWith(u8, t.key_ptr.*, ".weight_packed")) {
            const base = std.mem.cutSuffix(u8, t.key_ptr.*, ".weight_packed") orelse unreachable;
            const val = try Tensor.fromRegistry(r, base) orelse
                return error.MissingTensor;
            try tensors.append(a, val);
        }
    }
    return try tensors.toOwnedSlice(a);
}
