const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");

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

    var layer: st.File.NVFP4Tensor =
        f.getNVFP4Tensor("model.layers.0.self_attn.q_proj") orelse
        return error.MissingLayer;
    var reader = try layer.reader(allocator);
    defer reader.deinit();

    var out: [16]f32 = undefined;
    while (try reader.next(&out)) {
        for (out) |v| {
            try w.print("{} ", .{v});
        }
    }
    try w.print("\n", .{});
    try w.flush();
}
