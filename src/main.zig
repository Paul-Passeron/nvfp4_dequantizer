const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");
const job = @import("job");
const builtin = @import("builtin");

const usage =
    \\usage: nvfp4 [options] <input>
    \\
    \\Options:
    \\  -o, --output <path>           Output file     (default: out.safetensor)
    \\  -d, --datatype <f32|f16|bf16> Output type     (default: f32)
    \\  -h, --help                    Show this help
;

const Options = struct {
    in: []const u8,
    out: []const u8 = "out.safetensors",
    datatype: st.DataType = .F32,
};

const ParseError = error{
    MultipleInputs,
    MissingInput,
    UnkownOption,
    MissingValue,
    InvalidValue,
    HelpRequested,
} || std.mem.Allocator.Error;

fn splitEq(arg: []const u8) struct { name: []const u8, value: ?[]const u8 } {
    if (std.mem.startsWith(u8, arg, "--")) {
        if (std.mem.indexOfScalar(u8, arg, '=')) |i| {
            return .{ .name = arg[0..i], .value = arg[i + 1 ..] };
        }
    }
    return .{ .name = arg, .value = null };
}

fn isOption(
    name: []const u8,
    short: []const u8,
    long: []const u8,
) bool {
    return std.mem.eql(u8, name, short) or
        std.mem.eql(u8, name, long);
}

fn takeValue(
    args: []const [:0]const u8,
    i: *usize,
    inline_value: ?[]const u8,
    name: []const u8,
    err: *std.Io.Writer,
) ParseError![]const u8 {
    if (inline_value) |v| return v;
    if (i.* + 1 >= args.len) {
        failMessage(err, "error: {s} requires a value\n", .{name});
        return error.MissingValue;
    }
    i.* += 1;
    return args[i.*];
}

fn comptimeLowerCased(comptime s: []const u8) [s.len]u8 {
    var l: [s.len]u8 = undefined;
    for (s, &l) |c, *p| {
        p.* = std.ascii.toLower(c);
    }
    return l;
}

fn parseEnumLitMayBeLowercased(comptime t: type, from: []const u8) ?t {
    const info = @typeInfo(t);
    switch (info) {
        .@"enum" => {},
        else => @compileError("Expected an enum type"),
    }
    const t_info = comptime @typeInfo(t).@"enum";
    inline for (t_info.field_names, t_info.field_values) |name, value| {
        const lc = comptime comptimeLowerCased(name);
        if (std.mem.eql(u8, name, from) or
            std.mem.eql(u8, &lc, from))
            return @fromBackingInt(@intCast(value));
    }
    return null;
}

fn parseArgs(
    args: []const [:0]const u8,
    err: *std.Io.Writer,
) ParseError!Options {
    var opts: Options = .{ .in = "" };
    var i: usize = 1;
    var input_set = false;
    while (i < args.len) : (i += 1) {
        const raw = args[i];
        if (raw.len < 2 or raw[0] != '-') {
            if (input_set)
                return ParseError.MultipleInputs;
            input_set = true;
            opts.in = raw;
            continue;
        }

        const arg = splitEq(raw);
        if (isOption(arg.name, "-h", "--help")) {
            return ParseError.HelpRequested;
        } else if (isOption(arg.name, "-o", "--output")) {
            opts.out = try takeValue(args, &i, arg.value, arg.name, err);
        } else if (isOption(arg.name, "-d", "--datatype")) {
            const dt = try takeValue(args, &i, arg.value, arg.name, err);
            // Kind of over-engineered since we will only accept f32, f16
            // and bf16 but I wanted to try out zig comptime reflection
            // a bit.
            if (parseEnumLitMayBeLowercased(st.DataType, dt)) |val| {
                opts.datatype = val;
            } else {
                failMessage(err, "Invalid datatype `{s}`", .{dt});
                return ParseError.InvalidValue;
            }
        }
    }
    if (!input_set) return ParseError.MissingInput;
    return opts;
}

const Layout = struct {
    entries: []job.Entry,
    offsets: []usize,
    total_byte_len: usize,
    metadata: ?*const std.json.Value,

    fn fromRegistry(
        allocator: std.mem.Allocator,
        dt: st.DataType,
        r: *const st.Registry,
    ) !Layout {
        var entries: std.ArrayList(job.Entry) = .empty;
        var offsets: std.ArrayList(usize) = .empty;
        errdefer entries.deinit(allocator);
        errdefer offsets.deinit(allocator);

        const nvfp4s = try nvfp4.getAllNVFP4Tensors(allocator, dt, r);
        defer allocator.free(nvfp4s);

        const TMap = std.AutoHashMap(*const st.Tensor, void);
        var tensors: TMap = .init(allocator);
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
            try w.print("\"data_offsets\":[{}, {}],", .{
                offset,
                offset + e.byteLen(),
            });
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

    fn writeHeader(
        self: *const Layout,
        f_writer: *std.Io.File.Writer,
    ) !u64 {
        var fw = &f_writer.interface;
        // Placeholder header size
        _ = try fw.writeInt(u64, 0, .little);
        try self.writeJSONHeader(fw);

        const json_bytes: u64 = f_writer.logicalPos() - 8;
        const aligned = std.mem.alignForward(u64, json_bytes, 8);
        const padding: usize = @intCast(aligned - json_bytes);
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
        const jobs: []job.Job = try allocator.alloc(
            job.Job,
            self.entries.len,
        );

        for (self.entries, self.offsets, jobs) |e, offset, *j| {
            j.* = .{ .tensor = e, .dest = .{ .file = .{
                .buffer = try allocator.alloc(u8, 4 * 1024 * 1024),
                .file = out_file,
                .offset = offset + global_offset,
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

fn getRealPath(io: std.Io, path: []const u8, buf: []u8) ![]u8 {
    const a_f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer a_f.close(io);
    const length = try a_f.realPath(io, buf);
    return buf[0..length];
}

fn arePathsTheSame(io: std.Io, a: []const u8, b: []const u8) bool {
    var a_f_name_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var b_f_name_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const a_name = getRealPath(io, a, &a_f_name_buffer) catch return false;
    const b_name = getRealPath(io, b, &b_f_name_buffer) catch return false;
    return std.mem.eql(u8, a_name, b_name);
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

    const opts = parseArgs(args, err) catch |e| switch (e) {
        error.HelpRequested => {
            err.print(usage ++ "\n", .{}) catch {};
            try err.flush();
            return;
        },
        error.OutOfMemory => return e,
        else => fail(err),
    };

    switch (opts.datatype) {
        .BF16, .F16, .F32 => {},
        else => {
            failMessage(
                err,
                "Unsupported datatype `{}`, expected F32, F16 or BF16.",
                .{opts.datatype},
            );
            std.process.exit(2);
        },
    }

    try out.print("in:  `{s}`\n", .{opts.in});
    try out.print("out: `{s}`\n", .{opts.out});
    try out.print("dt: {}\n", .{opts.datatype});

    var r = st.Registry.fromPath(allocator, io, opts.in) catch |e| {
        failMessage(err, "could not load safetensors file at `{s}`: {}", .{
            opts.in,
            e,
        });
        fail(err);
    };
    defer r.deinit();

    var layout = try Layout.fromRegistry(allocator, opts.datatype, &r);
    defer layout.deinit(allocator);

    if (arePathsTheSame(io, opts.in, opts.out)) {
        failMessage(
            err,
            "Input and output files are the same: `{s}`",
            .{opts.in},
        );
        std.process.exit(2);
    }

    const out_file = try std.Io.Dir.cwd().createFile(io, opts.out, .{});
    defer out_file.close(io);

    var f_buffer: [32 * 1024]u8 = undefined;
    var f_writer = out_file.writer(io, &f_buffer);

    const header_bytes = try layout.writeHeader(&f_writer);
    const total_file_size = 8 + header_bytes + layout.total_byte_len;
    try out_file.setLength(io, total_file_size);

    const jobs = try layout.getJobs(allocator, &out_file, header_bytes + 8);
    defer deinitJobs(allocator, jobs);

    try job.runAllJobs(io, allocator, .{ .dequant = .simd }, jobs);

    try f_writer.flush();

    const end_time = clock.now(io);
    const duration = start_time.durationTo(end_time);

    try out.print("registry count = {}\n", .{r.count()});
    try out.print("jobs count = {}\n", .{jobs.len});
    try out.print("read \t{} bytes\nwrote \t{} bytes\nin {f}.\n", .{
        layout.total_byte_len,
        total_file_size,
        duration,
    });

    try out.flush();
}

fn failMessage(
    err: *std.Io.Writer,
    comptime fmt: []const u8,
    args: anytype,
) void {
    err.print("error: " ++ fmt ++ "\n", args) catch {};
    err.flush() catch {};
}

fn fail(err: *std.Io.Writer) noreturn {
    err.print(usage ++ "\n", .{}) catch {};
    std.process.exit(2);
}
