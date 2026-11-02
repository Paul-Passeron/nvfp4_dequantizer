const std = @import("std");

const usage =
    \\Usage: nvfp4 <in .safetensor> <out .safetensor>
;

const AllErrors = error{ InvalidArgCount, OutOfMemory };

const Options = struct {
    in: []u8,
    out: []u8,

    fn fromArgs(
        allocator: std.mem.Allocator,
        args: []const [:0]const u8,
    ) error{ InvalidArgCount, OutOfMemory }!Options {
        if (args.len != 3) {
            return error.InvalidArgCount;
        }
        var opts: Options = undefined;
        opts.in = try allocator.dupe(u8, args[1]);
        opts.out = try allocator.dupe(u8, args[2]);
        return opts;
    }

    fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        allocator.free(self.in);
        allocator.free(self.out);
    }
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

    var opts = Options.fromArgs(allocator, args) catch |e|
        failError(err, e);
    defer opts.deinit(allocator);

    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });

    try out.flush();
}

fn fail(err: *std.Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    err.print("error: " ++ fmt ++ "\n" ++ usage, args) catch {};
    err.flush() catch {};
    std.process.exit(2);
}

fn failError(err: *std.Io.Writer, e: AllErrors) noreturn {
    switch (e) {
        error.InvalidArgCount => {
            fail(err, "Invalid number of arguments, expected 2.", .{});
        },
        error.OutOfMemory => {
            fail(err, "Out of memory.", .{});
        },
    }
}
