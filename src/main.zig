const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");
const job = @import("job");

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
    std.Io.Dir.cwd().access(io, path, .{}) catch
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
        error.WriteFailed => blk: {
            err.print("error: Write failed\n", .{}) catch return error.UnknownIssue;

            break :blk error.IoProblem;
        },
        else => blk: {
            err.print("error: {}", .{e}) catch return error.UnknownIssue;
            break :blk error.UnknownIssue;
        },
    };
}

fn getAllNVFP4Tensors(
    a: std.mem.Allocator,
    r: *const st.Registry,
) ![]nvfp4.Tensor {
    var tensors = std.ArrayList(nvfp4.Tensor).empty;
    errdefer tensors.deinit(a);

    var it = r.iterator();
    while (it.next()) |t| {
        if (std.mem.endsWith(u8, t.key_ptr.*, ".weight_packed")) {
            const base = std.mem.cutSuffix(u8, t.key_ptr.*, ".weight_packed") orelse unreachable;
            const val = try nvfp4.Tensor.fromRegistry(r, base) orelse
                return error.MissingTensor;
            try tensors.append(a, val);
        }
    }
    return try tensors.toOwnedSlice(a);
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

    var entries: std.ArrayList(job.Entry) = .empty;
    var offsets: std.ArrayList(usize) = .empty;
    defer entries.deinit(allocator);
    defer offsets.deinit(allocator);

    const nvfp4s = try getAllNVFP4Tensors(allocator, &r);
    defer allocator.free(nvfp4s);

    var bases = try allocator.alloc([]const u8, nvfp4s.len);
    defer allocator.free(bases);

    var total_byte_len: usize = 0;

    for (nvfp4s, 0..) |t, i| {
        const name = t.weight_packed.name;
        const base = std.mem.cutSuffix(u8, name, ".weight_packed").?;
        bases[i] = base;
        try entries.append(allocator, .{ .nvfp4 = t });
        try offsets.append(allocator, total_byte_len);
        total_byte_len += t.dequantByteLen();
    }

    var it = r.iterator();

    while (it.next()) |t| {
        var is_nvfp4 = false;
        for (bases) |b| {
            if (std.mem.startsWith(u8, t.value_ptr.name, b)) {
                is_nvfp4 = true;
                break;
            }
        }
        if (is_nvfp4) continue;
        try entries.append(allocator, .{ .regular = t.value_ptr });
        try offsets.append(allocator, total_byte_len);
        total_byte_len += t.value_ptr.byte_len;
    }

    // TODO: maybe write to file directly, as it will save the cost of copying
    // the buffer to the file

    const output_buffer = try allocator.alloc(u8, total_byte_len);
    defer allocator.free(output_buffer);

    var jobs: []job.Job = try allocator.alloc(job.Job, entries.items.len);
    defer allocator.free(jobs);

    for (entries.items, 0..) |e, i| {
        const start = offsets.items[i];
        const end = start + e.byteLen();
        const mem: []u8 = output_buffer[start..end];
        jobs[i] = .{ .tensor = e, .dest = .{ .mem = mem } };
    }

    const clock = std.Io.Clock.awake;
    const start_time = clock.now(io);

    try job.runAllJobs(io, .{ .ordering = .simd }, jobs);

    const out_file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer out_file.close(io);

    var f_buffer: [32 * 1024]u8 = undefined;
    var f_writer = out_file.writer(io, &f_buffer);
    var fw = &f_writer.interface;

    // Placeholder header size
    const zero: u64 = 0;
    _ = try fw.write(std.mem.asBytes(&zero));

    try fw.print("{{", .{});

    for (entries.items, offsets.items, 0..) |e, offset, i| {
        if (i > 0) try fw.print(",", .{});
        try fw.print("\"{s}{s}\":{{", .{ e.name(), switch (e) {
            .nvfp4 => ".weight",
            else => "",
        } });
        try fw.print("\"dtype\":\"{s}\"", .{e.dtype().str()});
        try fw.print("\"offsets\":[{}, {}],", .{ offset, offset + e.byteLen() });
        try fw.print("\"shape\":[", .{});
        var shape_buffer: [2]u64 = undefined;
        const shape: []const u64 =
            switch (e) {
                .nvfp4 => |t| blk: {
                    shape_buffer = t.shape();
                    break :blk &shape_buffer;
                },
                .regular => |t| t.shape,
            };
        for (shape, 0..) |dim, dim_idx| {
            if (dim_idx > 0) try fw.print(",", .{});
            try fw.print("{}", .{dim});
        }
        try fw.print("]}}", .{});
    }
    try fw.print("}}", .{});

    const header_bytes: u64 = f_writer.pos - 8;
    try f_writer.seekTo(0);
    _ = try fw.writeInt(u64, header_bytes, .little);
    try f_writer.end();
    var rest: []const u8 = output_buffer;
    while (rest.len > 0) {
        const n = @min(rest.len, 1 << 30);
        try fw.writeAll(rest[0..n]);
        rest = rest[n..];
    }
    try fw.flush();

    const end_time = clock.now(io);
    const duration = start_time.durationTo(end_time);

    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });
    try out.print("registry count = {}\n", .{r.count()});
    try out.print("jobs count = {}\n", .{jobs.len});
    try out.print("Read {} MB and wrote {} MB in {f}.\n", .{ total_byte_len / 1024 / 1024, f_writer.pos / 1024 / 1024, duration });
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

    doWork(allocator, io, out, args) catch |e|
        failError(err, sanitizeError(e, err));

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
