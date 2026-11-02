const std = @import("std");

const usage =
    \\Usage: nvfp4 <in .safetensor> <out .safetensor>
;

const Options = struct {
    in: []u8,
    out: []u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [4096]u8 = undefined;
    var out_writer = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_writer.interface;

    var err_buf: [1024]u8 = undefined;
    var err_writer = std.Io.File.stderr().writer(io, &err_buf);
    const err = &err_writer.interface;

    if (args.len != 3) fail(
        err,
        "Wrong number of arguments, expected 2 but got {}",
        .{args.len - 1},
    );

    const opts = blk: {
        var opts: Options = undefined;
        opts.in = try allocator.dupe(u8, args[1]);
        opts.out = try allocator.dupe(u8, args[2]);
        break :blk opts;
    };
    defer allocator.free(opts.in);
    defer allocator.free(opts.out);

    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });

    try out.flush();
}

fn fail(err: *std.Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    err.print("error: " ++ fmt ++ "\n" ++ usage, args) catch {};
    err.flush() catch {};
    std.process.exit(2);
}
