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

    var f = try st.File.open(io, allocator, "nvfp4/model.safetensors");
    defer f.close();

    var converted_f = try st.File.open(io, allocator, "converted_f32/model-00001-of-00001.safetensors");
    defer converted_f.close();

    var it = converted_f.iterator();
    while (it.next()) |val| {
        try w.print("{s} {}\n", .{ val.name, val.dtype });
    }

    var layer = try nvfp4.NVFP4Tensor.from(&f, "model.layers.0.self_attn.q_proj") orelse
        return error.MissingLayer;
    var reader = try layer.reader(allocator);
    defer reader.deinit();

    const t = (try converted_f.tensor(f32, "model.layers.0.self_attn.q_proj.weight")) orelse
        return error.MissingLayer;

    var c_reader = try t.reader(allocator);
    defer c_reader.deinit();

    var out: [16]f32 = undefined;
    var converted_out: [16]f32 = undefined;

    while (try reader.next(&out)) {
        for (&converted_out) |*cv| {
            if (try c_reader.next()) |x| {
                cv.* = x;
            } else {
                break;
            }
        }
        for (out, converted_out) |v, cv| {
            try w.print("({} {}) ", .{ v, cv });
        }
    }
    try w.print("\n", .{});
    try w.flush();
}
