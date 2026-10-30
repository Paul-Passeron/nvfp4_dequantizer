const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const Io = std.Io;
const Clock = Io.Clock;
// const print = std.debug.print;

test "nvfp4 and f32 are close enough" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var f = try st.File.open(io, a, "nvfp4/model.safetensors");
    defer f.close();

    var converted_f = try st.File.open(io, a, "converted_f32/model-00001-of-00001.safetensors");
    defer converted_f.close();

    var t = try nvfp4.NVFP4Tensor.from(
        &f,
        "model.layers.0.self_attn.q_proj",
    ) orelse return error.MissingLayer;
    var reader = try t.reader(a);
    defer reader.deinit();

    const ct = (try converted_f.tensor(
        f32,
        "model.layers.0.self_attn.q_proj.weight",
    )) orelse return error.MissingLayer;

    var c_reader = try ct.reader(a);
    defer c_reader.deinit();

    var out: [16]f32 = undefined;
    var converted_out: [16]f32 = undefined;

    while (true) {
        if (!try reader.next(&out)) break;
        for (&converted_out) |*cv| {
            const next = try c_reader.next();
            try std.testing.expect(next != null);
            cv.* = next.?;
        }
        for (out, converted_out) |actual, expected| {
            try std.testing.expectApproxEqRel(expected, actual, 1e-3);
        }
    }
}

fn benchmark(
    path: []const u8,
    base: []const u8,
    allocator: std.mem.Allocator,
) !Io.Duration {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try st.File.open(io, allocator, path);
    defer f.close();

    var t = try nvfp4.NVFP4Tensor.from(&f, base) orelse
        return error.MissingLayer;
    var reader = try t.reader(allocator);
    defer reader.deinit();

    var out: [16]f32 = undefined;

    const start = Clock.real.now(io);
    {
        while (try reader.next(&out)) {}
    }
    const end = Clock.real.now(io);

    return start.durationTo(end);
}

fn nextFloat(r: *std.Io.Reader) !?f32 {
    const bits = r.takeInt(u32, .little) catch |err| switch (err) {
        error.EndOfStream => return null,
        else => |e| return e,
    };
    return @bitCast(bits);
}

fn benchmark_with_reader_one_at_a_time(
    path: []const u8,
    base: []const u8,
    allocator: std.mem.Allocator,
) !Io.Duration {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try st.File.open(io, allocator, path);
    defer f.close();

    var t = try nvfp4.NVFP4Tensor.from(&f, base) orelse
        return error.MissingLayer;
    var blocks = try t.reader(allocator);
    defer blocks.deinit();

    const buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(buffer);

    const start = Clock.real.now(io);
    {
        var reader = nvfp4.Reader.init(blocks, buffer);
        const r = &reader.interface;
        while (try nextFloat(r)) |_| {}
    }
    const end = Clock.real.now(io);

    return start.durationTo(end);
}

fn benchmark_with_reader_into_writer(
    path: []const u8,
    base: []const u8,
    allocator: std.mem.Allocator,
    w: *std.Io.Writer,
) !Io.Duration {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try st.File.open(io, allocator, path);
    defer f.close();

    var t = try nvfp4.NVFP4Tensor.from(&f, base) orelse
        return error.MissingLayer;
    var blocks = try t.reader(allocator);
    defer blocks.deinit();

    const buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(buffer);

    const start = Clock.real.now(io);
    {
        var reader = nvfp4.Reader.init(blocks, buffer);
        const r = &reader.interface;
        _ = try r.streamRemaining(w);
    }
    const end = Clock.real.now(io);

    return start.durationTo(end);
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const filename = "nvfp4/model.safetensors";
    const tensor = "model.layers.0.self_attn.q_proj";

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    std.debug.print("`benchmark` took {f}\n", .{
        try benchmark(filename, tensor, allocator),
    });

    std.debug.print("`benchmark_with_reader_one_at_a_time` took {f}\n", .{
        try benchmark_with_reader_one_at_a_time(
            filename,
            tensor,
            allocator,
        ),
    });

    std.debug.print(
        "`benchmark_with_reader_into_writer` took {f}\n",
        .{try benchmark_with_reader_into_writer(
            filename,
            tensor,
            allocator,
            &aw.writer,
        )},
    );
}
