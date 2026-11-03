const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");
const job = @import("job");

const usage =
    \\usage: nvfp4 <in .safetensor> <out .safetensor>
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

        var tensors: std.AutoHashMap(*const st.Tensor, void) = .init(allocator);
        defer tensors.deinit();

        var total_byte_len: usize = 0;

        for (nvfp4s) |t| {
            try tensors.put(t.weight_global_scale, {});
            try tensors.put(t.weight_packed, {});
            try tensors.put(t.weight_scale, {});
            try entries.append(allocator, .{ .nvfp4 = t });
            try offsets.append(allocator, total_byte_len);
            total_byte_len += t.dequantByteLen();
        }

        var it = r.iterator();

        while (it.next()) |t| {
            if (tensors.contains(t.value_ptr)) continue;
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

    fn writeHeader(self: *const Layout, f_writer: *std.Io.File.Writer) !u64 {
        var fw = &f_writer.interface;
        // Placeholder header size
        _ = try fw.writeInt(u64, 0, .little);
        try self.writeJSONHeader(fw);

        const json_bytes: u64 = f_writer.logicalPos() - 8;
        const padding: usize =
            @intCast(std.mem.alignForward(u64, json_bytes, 8) - json_bytes);
        try fw.splatByteAll(' ', padding);

        const header_bytes: u64 = json_bytes + padding;
        try f_writer.seekTo(0);
        _ = try fw.writeInt(u64, header_bytes, .little);
        return header_bytes;
    }

    fn getJobs(
        self: *Layout,
        allocator: std.mem.Allocator,
        out_file: *const std.Io.File,
        global_offset: u64,
    ) ![]job.Job {
        var jobs: []job.Job = try allocator.alloc(job.Job, self.entries.len);

        for (self.entries, 0..) |e, i| {
            jobs[i] = .{ .tensor = e, .dest = .{ .file = .{
                .buffer = try allocator.alloc(u8, 4 * 1024 * 1024),
                .file = out_file,
                .offset = self.offsets[i] + global_offset,
            } } };
        }

        return jobs;
    }
};

fn deinitJobs(allocator: std.mem.Allocator, jobs: []job.Job) void {
    for (jobs) |j| {
        switch (j.dest) {
            .file => |f| {
                allocator.free(f.buffer);
            },
            else => {},
        }
    }
    allocator.free(jobs);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [4096]u8 = undefined;
    var out_writer = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const out = &out_writer.interface;

    var err_buf: [4096]u8 = undefined;
    var err_writer = std.Io.File.stderr().writerStreaming(io, &err_buf);
    const err = &err_writer.interface;

    const clock = std.Io.Clock.awake;
    const start_time = clock.now(io);

    const opts = Options.fromArgs(args) orelse return error.CouldntParseArgs;

    var r = st.Registry.fromPath(allocator, io, opts.in) catch |e|
        fail(err, "could not load safetensors file at `{s}`: {}", .{ opts.in, e });
    defer r.deinit();

    var layout = try Layout.fromRegistry(allocator, &r);
    defer layout.deinit(allocator);

    const out_file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer out_file.close(io);

    var f_buffer: [32 * 1024]u8 = undefined;
    var f_writer = out_file.writer(io, &f_buffer);

    const header_bytes = try layout.writeHeader(&f_writer);
    const total_file_size = 8 + header_bytes + layout.total_byte_len;
    try out_file.setLength(io, total_file_size);

    const jobs = try layout.getJobs(allocator, &out_file, header_bytes + 8);
    defer deinitJobs(allocator, jobs);

    try job.runAllJobs(io, .{ .ordering = .simd }, jobs);

    try f_writer.flush();

    const end_time = clock.now(io);
    const duration = start_time.durationTo(end_time);

    try out.print("in = `{s}`, out = `{s}`\n", .{ opts.in, opts.out });
    try out.print("registry count = {}\n", .{r.count()});
    try out.print("jobs count = {}\n", .{jobs.len});
    try out.print("read \t{} bytes\nwrote \t{} bytes\nin {f}.\n", .{
        layout.total_byte_len,
        total_file_size,
        duration,
    });

    try out.flush();
}

fn fail(err: *std.Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    err.print("error: " ++ fmt ++ "\n" ++ usage ++ "\n", args) catch {};
    err.flush() catch {};
    std.process.exit(2);
}
