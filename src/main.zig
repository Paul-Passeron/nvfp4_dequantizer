const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");
const job = @import("job");

const usage =
    \\Usage: nvfp4 <in .safetensor> <out .safetensor>
;

const Options = struct {
    in: []const u8,
    out: []const u8,

    fn fromArgs(args: []const [:0]const u8) ?Options {
        if (args.len != 3) {
            return null;
        }
        return .{
            .in = args[1],
            .out = args[2],
        };
    }
};

fn checkFile(io: std.Io, path: []const u8) error{FileNotFound}!void {
    std.Io.Dir.cwd().access(io, path, .{}) catch
        return error.FileNotFound;
}

pub inline fn errorSet(comptime Err: type, err: anytype) bool {
    return @hasField(std.meta.FieldEnum(Err), @errorName(err));
}

const Layout = struct {
    entries: []job.Entry,
    offsets: []usize,
    total_byte_len: usize,
    metadata: ?*const std.json.Value,

    fn fromRegistry(
        allocator: std.mem.Allocator,
        r: *const st.Registry,
    ) !Layout {
        var entries: std.ArrayList(job.Entry) = .empty;
        var offsets: std.ArrayList(usize) = .empty;
        errdefer entries.deinit(allocator);
        errdefer offsets.deinit(allocator);

        const nvfp4s = try nvfp4.getAllNVFP4Tensors(allocator, r);
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
                const name = t.value_ptr.name;
                if (name.len > b.len and
                    std.mem.startsWith(u8, name, b) and
                    name[b.len] == '.')
                {
                    is_nvfp4 = true;
                    break;
                }
            }
            if (is_nvfp4) continue;
            try entries.append(allocator, .{ .regular = t.value_ptr });
            try offsets.append(allocator, total_byte_len);
            total_byte_len += t.value_ptr.byte_len;
        }
        return .{
            .entries = try entries.toOwnedSlice(allocator),
            .offsets = try offsets.toOwnedSlice(allocator),
            .total_byte_len = total_byte_len,
            .metadata = if (r.metadata) |_| &r.metadata.? else null,
        };
    }

    fn deinit(self: *Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.offsets);
    }

    fn writeJSONHeader(
        self: *const Layout,
        w: *std.Io.Writer,
    ) !void {
        try w.print("{{", .{});

        for (self.entries, self.offsets, 0..) |e, offset, i| {
            if (i > 0) try w.print(",", .{});
            try w.print("\"{s}{s}\":{{", .{ e.name(), switch (e) {
                .nvfp4 => ".weight",
                else => "",
            } });
            try w.print("\"dtype\":\"{s}\",", .{e.dtype().str()});
            try w.print("\"data_offsets\":[{}, {}],", .{ offset, offset + e.byteLen() });
            try w.print("\"shape\":[", .{});
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
                if (dim_idx > 0) try w.print(",", .{});
                try w.print("{}", .{dim});
            }
            try w.print("]}}", .{});
        }
        if (self.metadata) |m| {
            try w.print(",\"__metadata__\":", .{});
            try std.json.Stringify.value(m.*, .{
                .whitespace = .minified,
            }, w);
        }
        try w.print("}}", .{});
    }
};

fn doWork(
    allocator: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    args: []const [:0]const u8,
) !void {
    const opts = Options.fromArgs(args) orelse return error.CouldntParseArgs;

    try checkFile(io, opts.in);

    var r: st.Registry = try .fromPath(allocator, io, opts.in);
    defer r.deinit();

    var layout = try Layout.fromRegistry(allocator, &r);
    defer layout.deinit(allocator);

    // TODO: maybe write to file directly, as it will save the cost of copying
    // the buffer to the file

    const out_file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer out_file.close(io);

    var f_buffer: [32 * 1024]u8 = undefined;
    var f_writer = out_file.writer(io, &f_buffer);
    var fw = &f_writer.interface;

    // Placeholder header size
    const zero: u64 = 0;
    _ = try fw.write(std.mem.asBytes(&zero));
    try layout.writeJSONHeader(fw);

    const json_bytes: u64 = f_writer.logicalPos() - 8;
    const padding: usize =
        @intCast(std.mem.alignForward(u64, json_bytes, 8) - json_bytes);
    try fw.splatByteAll(' ', padding);

    const header_bytes: u64 = json_bytes + padding;
    try f_writer.seekTo(0);
    _ = try fw.writeInt(u64, header_bytes, .little);
    try fw.flush();

    var jobs: []job.Job = try allocator.alloc(job.Job, layout.entries.len);
    defer allocator.free(jobs);

    for (layout.entries, 0..) |e, i| {
        jobs[i] = .{ .tensor = e, .dest = .{ .file = .{
            .file = &out_file,
            .offset = layout.offsets[i] + header_bytes + 8,
        } } };
    }

    const clock = std.Io.Clock.awake;
    const start_time = clock.now(io);

    try job.runAllJobs(io, .{ .ordering = .simd }, jobs);

    const end_time = clock.now(io);
    const duration = start_time.durationTo(end_time);

    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });
    try out.print("registry count = {}\n", .{r.count()});
    try out.print("jobs count = {}\n", .{jobs.len});
    const total_file_size = 8 + header_bytes + layout.total_byte_len;
    try out.print("read \t{} bytes\nwrote \t{} bytes\nin {f}.\n", .{ layout.total_byte_len, total_file_size, duration });
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

    doWork(allocator, io, out, args) catch |e| fail(err, "{}", .{e});

    try out.flush();
}

fn fail(err: *std.Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    err.print("error: " ++ fmt ++ "\n" ++ usage, args) catch {};
    err.flush() catch {};
    std.process.exit(2);
}
