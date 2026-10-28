const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");

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

const FOUR_KB = 4 * 1024;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var write_buf: [FOUR_KB]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &write_buf);
    const w = &fw.interface;

    // const filename = "converted/model-00001-of-00001.safetensors";
    const filename = "nvfp4/model.safetensors";
    var file = try std.Io.Dir.cwd().openFile(io, filename, .{ .mode = .read_only });
    defer file.close(io);
    var read_buf: [FOUR_KB]u8 = undefined;
    var f_reader: std.Io.File.Reader = file.reader(io, &read_buf);

    const n = try f_reader.interface.takeInt(u64, .little);
    try w.print("N is {}\n", .{n});

    const json_blob = try f_reader.interface.readAlloc(allocator, n);
    defer allocator.free(json_blob);

    var header = try st.SafeTensorHeader.fromJSON(allocator, json_blob);
    defer header.deinit();

    const M = std.AutoHashMap(st.DataType, void);
    var m: M = .init(allocator);
    defer m.deinit();

    for (0.., header.tensors) |i, t| {
        if (!std.mem.startsWith(u8, t.name, "model.layers.0.self_attn.q_proj")) {
            continue;
        }
        try w.print("{s}: {}\n", .{ t.name, i });
        // if (std.mem.endsWith(u8, t.name, "weight_packed")) {
        //     try w.print("{s}: {}\n", .{ t.name, i });
        // }
        if (t.info.dtype != .U8) {
            continue;
        }
        try w.print("u8 at {}\n", .{i});
        try f_reader.seekTo(t.info.data_offsets[0]);
        const len = t.info.data_offsets[1] - t.info.data_offsets[0];
        const r = &f_reader.interface;
        for (0..len) |_| {
            const b = try r.takeByte();
            const lo, const hi = nvfp4.unpack(b);
            const lo_f = nvfp4.decodeE2M1(lo);
            const hi_f = nvfp4.decodeE2M1(hi);
            try w.print("{}: {} and {}\n", .{ b, lo_f, hi_f });
        }
    }

    var out: [16]f32 = undefined;
    const scale: u8 = 2;
    const global: f32 = 1.0;
    const bytes: [16]u4 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    nvfp4.dequantBlock(@ptrCast(&bytes), scale, global, &out);
    for (out) |o| {
        try w.print("{}, ", .{o});
    }
    try w.print("\n", .{});
    try w.flush();
}
