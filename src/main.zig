const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");
const job = @import("job");
const builtin = @import("builtin");

const usage_message =
    \\usage: nvfp4 [options] <input>
    \\
    \\Options:
    \\  -o, --output <path>           Output file                       (default: out.safetensors)
    \\  -d, --datatype <f32|f16|bf16> Output type                       (default: f32)
    \\      --e2m1 <tbl|lut|bits>     Set e2m1 dequantization algorithm (default: tbl on aarch64, lut on other platforms)
    \\      --e4m3 <lut|bits>         Set e4m3 dequantization algorithm (default: lut)
    \\      --dequant <simd|seq>      Set dequantization parallelism    (default: simd)
    \\  -h, --help                    Show this help
;

const Options = struct {
    in: []const u8,
    out: []const u8 = "out.safetensors",
    datatype: st.DataType = .F32,
    dequant_algo: nvfp4.DequantAlgo = .simd,
    e2m1: nvfp4.e2m1Alg = nvfp4.default_e2m1_algo,
    e4m3: nvfp4.e4m3Alg = nvfp4.default_e4m3_algo,

    fn config(self: Options) nvfp4.DequantConfig {
        return .{
            .dequant = self.dequant_algo,
            .e2m1 = self.e2m1,
            .e4m3 = self.e4m3,
        };
    }
};

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
    short: ?[]const u8,
    long: []const u8,
) bool {
    if (short) |s|
        if (std.mem.eql(u8, name, s))
            return true;
    return std.mem.eql(u8, name, long);
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
        self: *const Layout,
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

fn fileSize(io: std.Io, path: []const u8) !u64 {
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    return stat.size;
}

const Cli = struct {
    io: std.Io,
    allocator: std.mem.Allocator,

    out_buf: []u8,
    out_writer: std.Io.File.Writer,

    err_buf: []u8,
    err_writer: std.Io.File.Writer,

    fn init(from: std.process.Init) !Cli {
        const allocator = from.gpa;
        const io = from.io;

        const out_buf = try allocator.alloc(u8, 1024 * 1024);
        errdefer allocator.free(out_buf);
        const out_writer = std.Io.File.stdout().writerStreaming(io, out_buf);

        const err_buf = try allocator.alloc(u8, 1024 * 1024);
        errdefer allocator.free(err_buf);
        const err_writer = std.Io.File.stderr().writerStreaming(io, err_buf);

        return .{
            .io = io,
            .allocator = allocator,
            .out_buf = out_buf,
            .out_writer = out_writer,
            .err_buf = err_buf,
            .err_writer = err_writer,
        };
    }

    fn deinit(self: *Cli) void {
        self.err_writer.flush() catch {};
        self.out_writer.flush() catch {};

        self.allocator.free(self.err_buf);
        self.allocator.free(self.out_buf);
    }

    fn print(
        self: *Cli,
        comptime fmt: []const u8,
        args: anytype,
    ) !void {
        try self.out_writer.interface.print(fmt, args);
        try self.out_writer.flush();
    }

    fn usage(
        self: *Cli,
    ) !void {
        try self.err_writer.interface.print(usage_message ++ "\n", .{});
        try self.err_writer.flush();
    }

    fn fail(
        self: *Cli,
        comptime fmt: []const u8,
        args: anytype,
    ) anyerror {
        try self.err_writer.interface.print("error: " ++ fmt ++ "\n", args);
        try self.err_writer.flush();
        return error.CloseProgramWithError;
    }

    fn failWithUsage(
        self: *Cli,
        comptime fmt: []const u8,
        args: anytype,
    ) anyerror {
        try self.err_writer.interface.print("error: " ++ fmt ++ "\n", args);
        try self.err_writer.flush();
        try self.usage();
        return error.CloseProgramWithError;
    }

    fn takeValue(
        self: *Cli,
        args: []const [:0]const u8,
        i: *usize,
        inline_value: ?[]const u8,
        name: []const u8,
    ) ![]const u8 {
        if (inline_value) |v| return v;
        if (i.* + 1 >= args.len) {
            const err = self.fail("{s} requires a value", .{name});
            try self.usage();
            return err;
        }
        i.* += 1;
        return args[i.*];
    }

    fn validateOpts(self: *Cli, opts: Options) !void {
        switch (opts.datatype) {
            .BF16, .F16, .F32 => {},
            else => return self.fail(
                "unsupported datatype `{}`, expected F32, F16 or BF16.",
                .{opts.datatype},
            ),
        }

        if (arePathsTheSame(self.io, opts.in, opts.out)) {
            return self.fail(
                "input and output files are the same: `{s}`",
                .{opts.in},
            );
        }
    }

    fn getOpts(self: *Cli, args: []const [:0]const u8) !Options {
        const opts = try self.parseArgs(args);
        try self.validateOpts(opts);
        return opts;
    }

    fn parseArgs(self: *Cli, args: []const [:0]const u8) !Options {
        var opts: Options = .{ .in = "" };
        var i: usize = 1;
        var input_set = false;
        while (i < args.len) : (i += 1) {
            const raw = args[i];
            if (raw.len < 2 or raw[0] != '-') {
                if (input_set)
                    return self.fail("too many input file paths", .{});
                input_set = true;
                opts.in = raw;
                continue;
            }

            const arg = splitEq(raw);
            if (isOption(arg.name, "-h", "--help")) {
                return error.HelpRequested;
            } else if (isOption(arg.name, "-o", "--output")) {
                opts.out = try self.takeValue(args, &i, arg.value, arg.name);
            } else if (isOption(arg.name, null, "--e2m1")) {
                const val = try self.takeValue(args, &i, arg.value, arg.name);
                if (std.mem.eql(u8, val, "lut")) {
                    opts.e2m1 = .look_up_table;
                } else if (std.mem.eql(u8, val, "tbl")) {
                    opts.e2m1 = .tbl;
                } else if (std.mem.eql(u8, val, "bits")) {
                    opts.e2m1 = .bits;
                } else {
                    return self.failWithUsage("Invalid e2m1 algorithm `{s}`", .{val});
                }
            } else if (isOption(arg.name, null, "--e4m3")) {
                const val = try self.takeValue(args, &i, arg.value, arg.name);
                if (std.mem.eql(u8, val, "lut")) {
                    opts.e4m3 = .look_up_table;
                } else if (std.mem.eql(u8, val, "bits")) {
                    opts.e4m3 = .bits;
                } else {
                    return self.failWithUsage("Invalid e4m3 algorithm `{s}`", .{val});
                }
            } else if (isOption(arg.name, null, "--dequant")) {
                const val = try self.takeValue(args, &i, arg.value, arg.name);
                if (std.mem.eql(u8, val, "simd")) {
                    opts.dequant_algo = .simd;
                } else if (std.mem.eql(u8, val, "seq")) {
                    opts.dequant_algo = .sequential;
                } else {
                    return self.failWithUsage("Invalid dequantization algorithm `{s}`", .{val});
                }
            } else if (isOption(arg.name, "-d", "--datatype")) {
                const dt = try self.takeValue(args, &i, arg.value, arg.name);
                // Kind of over-engineered since we will only accept f32, f16
                // and bf16 but I wanted to try out zig comptime reflection
                // a bit.
                if (parseEnumLitMayBeLowercased(st.DataType, dt)) |val| {
                    opts.datatype = val;
                } else {
                    return self.failWithUsage("Invalid datatype `{s}`", .{dt});
                }
            } else {
                return self.failWithUsage("unknown option `{s}`", .{arg.name});
            }
        }
        if (!input_set)
            return self.failWithUsage("missing input file path", .{});
        return opts;
    }

    fn getRegistry(self: *Cli, opts: Options) !st.Registry {
        return st.Registry.fromPath(self.allocator, self.io, opts.in) catch |err| switch (err) {
            error.FileNotFound => self.fail("file not found: `{s}`", .{opts.in}),
            else => err,
        };
    }

    fn writeDequantizedFile(
        self: *Cli,
        opts: Options,
        out_file: *const std.Io.File,
        layout: *const Layout,
    ) !u64 {
        var f_buffer: [32 * 1024]u8 = undefined;
        var f_writer = out_file.writer(self.io, &f_buffer);

        const header_bytes = try layout.writeHeader(&f_writer);
        const total_header_bytes = 8 + header_bytes;
        const total_file_size = total_header_bytes + layout.total_byte_len;

        try out_file.setLength(self.io, total_file_size);

        const jobs = try layout.getJobs(self.allocator, out_file, total_header_bytes);
        defer deinitJobs(self.allocator, jobs);

        try job.runAllJobs(self.io, self.allocator, opts.config(), jobs);

        try f_writer.flush();

        return total_file_size;
    }

    fn runImpl(self: *Cli, args: []const [:0]const u8) !void {
        const opts = try self.getOpts(args);

        const input_file_size = try fileSize(self.io, opts.in);

        var r = try self.getRegistry(opts);
        defer r.deinit();

        var layout = try Layout.fromRegistry(self.allocator, opts.datatype, &r);
        defer layout.deinit(self.allocator);

        const clock = std.Io.Clock.awake;
        const start_time = clock.now(self.io);

        const out_file = try std.Io.Dir.cwd().createFile(
            self.io,
            opts.out,
            .{},
        );
        defer out_file.close(self.io);

        const total_file_size =
            try self.writeDequantizedFile(opts, &out_file, &layout);

        const duration = start_time.durationTo(clock.now(self.io));

        try self.print("in:  `{s}`\n", .{opts.in});
        try self.print("out: `{s}`\n", .{opts.out});
        try self.print("dt: {}\n", .{opts.datatype});
        try self.print("e2m1: {}\n", .{opts.e2m1});
        try self.print("e4m3: {}\n", .{opts.e4m3});
        try self.print("dequant: {}\n", .{opts.dequant_algo});
        try self.print("read  \t{} bytes\n", .{input_file_size});
        try self.print("wrote \t{} bytes\n", .{total_file_size});
        try self.print("took  \t{f}\n", .{duration});
    }

    fn run(self: *Cli, args: []const [:0]const u8) !void {
        self.runImpl(args) catch |err| switch (err) {
            error.HelpRequested => try self.usage(),
            error.CloseProgramWithError => return err,
            else => return self.fail("{}", .{err}),
        };
    }
};

pub fn mainImpl(init: std.process.Init) !u8 {
    var cli: Cli = try .init(init);
    defer cli.deinit();

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    cli.run(args) catch |err| switch (err) {
        error.CloseProgramWithError => return 2,
        else => unreachable,
    };

    return 0;
}

pub fn main(init: std.process.Init) !void {
    std.process.exit(try mainImpl(init));
}
