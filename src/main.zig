const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const Io = std.Io;
const Clock = Io.Clock;
const print = std.debug.print;

fn print_names(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    var reg: st.Registry =
        try .fromPath(allocator, io, path);
    defer reg.deinit();
    var it = reg.iterator();
    while (it.next()) |t| {
        print("{s}\n", .{t.value_ptr.name});
    }
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const filename = "nvfp4/model.safetensors";

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    try print_names(
        allocator,
        io,
        filename,
    );
    print("\n\n", .{});
    try print_names(
        allocator,
        io,
        "converted_f32/model-00001-of-00001.safetensors",
    );
}
