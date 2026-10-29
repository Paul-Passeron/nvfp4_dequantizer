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

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var write_buf: [4096]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &write_buf);
    const w = &fw.interface;

    // const filename = "converted/model-00001-of-00001.safetensors";
    const filename = "nvfp4/model.safetensors";
    var f = try st.File.open(io, allocator, filename);
    defer f.close();
    //
    //    var
    //
    //     var count: u32 = 0;
    //
    //     var it = f.iterator();
    //     while (it.next()) |t| {
    //         if (!std.mem.startsWith(u8, t.name, "model.layers.0.self_attn.q_proj")) {
    //             continue;
    //         }
    //
    //         if (std.mem.endsWith(u8, t.name, "input_global_scale")) {
    //             input_global_scale = t;
    //             count += 1;
    //         }
    //         if (std.mem.endsWith(u8, t.name, "weight_global_scale")) {
    //             weight_global_scale = t;
    //             count += 1;
    //         }
    //         if (std.mem.endsWith(u8, t.name, "weight_scale")) {
    //             weight_scale = t;
    //             count += 1;
    //         }
    //         if (std.mem.endsWith(u8, t.name, "weight_packed")) {
    //             weight_packed = t;
    //             count += 1;
    //         }
    //     }
    //
    //     if (count != 4) {
    //         try w.print("Count = {d}", .{count});
    //     }

    const layer: st.File.NVFP4Layer = try (f.getNVFP4Layer("model.layers.0.self_attn.q_proj") orelse error.MissingLayer);

    // Make sure the length are as expected

    // try std.testing.expectEqual(layer.weight_global_scale.byte_len * 8 / 32, 1);
    // try std.testing.expectEqual(
    //     layer.weight_scale.byte_len * 16 * 4 / 8,
    //     layer.weight_packed.byte_len,
    // );

    const block_scales = try f.readBytes(allocator, layer.weight_scale);
    defer allocator.free(block_scales);
    const global_scale_array = try f.readBytes(allocator, layer.weight_global_scale);
    const global_scale: f32 = @bitCast(std.mem.readInt(u32, global_scale_array[0..4], .little));
    allocator.free(global_scale_array);

    const tensors = try f.readBytes(allocator, layer.weight_packed);
    defer allocator.free(tensors);

    var out: [16]f32 = undefined;
    for (block_scales, 0..) |scale, i| {
        const bytes = tensors[i * 8 ..][0..8];
        nvfp4.dequantBlock(bytes, scale, global_scale, &out);
        for (out) |v| {
            try w.print("{} ", .{v});
        }
        try w.print("\n", .{});
    }
    try w.flush();
}
