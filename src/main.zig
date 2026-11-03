const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");

const usage =
    \\Usage: nvfp4 <in .safetensor> <out .safetensor>
;

const AllErrors = error{
    InvalidArgCount,
    OutOfMemory,
    InputFileNotFound,
    IoProblem,
    UnknownIssue,
};

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

fn checkFile(io: std.Io, path: []const u8) error{FileNotFound}!void {
    (if (std.mem.startsWith(u8, path, "/"))
        std.Io.Dir.accessAbsolute(io, path, .{})
    else
        std.Io.Dir.access(std.Io.Dir.cwd(), io, path, .{})) catch
        return error.FileNotFound;
}

pub inline fn errorSet(comptime Err: type, err: anytype) bool {
    return @hasField(std.meta.FieldEnum(Err), @errorName(err));
}

fn sanitizeError(e: anyerror, err: *std.Io.Writer) AllErrors {
    return switch (e) {
        error.InvalidArgCount => error.InvalidArgCount,
        error.OutOfMemory => error.OutOfMemory,
        error.InputFileNotFound => error.InputFileNotFound,
        error.IoProblem => error.IoProblem,
        error.UnknownIssue => error.UnknownIssue,
        error.FileNotFound => error.InputFileNotFound,
        error.WriteFailed => error.IoProblem,
        else => blk: {
            err.print("error: {}", .{e}) catch return error.UnknownIssue;
            break :blk error.UnknownIssue;
        },
    };
}

fn doWork(
    allocator: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    args: []const [:0]const u8,
) !void {
    var opts = try Options.fromArgs(allocator, args);
    defer opts.deinit(allocator);
    try checkFile(io, opts.in);
    var r: st.Registry = try .fromPath(allocator, io, opts.in);
    defer r.deinit();

    try out.print("registry count = {}\n", .{r.count()});
    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });
}

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

    doWork(allocator, io, out, args) catch |e| {
        failError(err, sanitizeError(e, err));
    };

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
        error.InputFileNotFound => {
            fail(err, "Input file not found", .{});
        },
        error.IoProblem => {
            fail(err, "Io problem", .{});
        },
        error.UnknownIssue => {
            fail(err, "Unknown issue", .{});
        },
    }
}
