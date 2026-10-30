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

fn benchmark(
    a: std.mem.Allocator,
    io: std.Io,
    filename: []const u8,
    base: []const u8,
) !std.Io.Duration {
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    const t = try nvfp4.Tensor.fromRegistry(&reg, base) orelse
        return error.MissingTensor;

    const start = std.Io.Clock.real.now(io);

    var packed_buffer: [4096]u8 = undefined;
    var scales_buffer: [4096]u8 = undefined;
    var reader = try t.reader(io, .{
        .packed_bytes = &packed_buffer,
        .scales = &scales_buffer,
    });
    defer reader.deinit();

    _ = try reader.interface.streamRemaining(&aw.writer);

    const end = std.Io.Clock.real.now(io);
    return start.durationTo(end);
}

pub fn main() !void {
    const a = std.heap.c_allocator;
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    print("benchmark took {f}\n", .{try benchmark(
        a,
        io,
        "nvfp4/model.safetensors",
        "model.layers.0.self_attn.q_proj",
    )});
}
