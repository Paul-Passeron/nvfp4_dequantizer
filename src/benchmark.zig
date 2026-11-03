const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const job = @import("job");
const Io = std.Io;
const Clock = Io.Clock;
const print = std.debug.print;

const BenchRes = struct {
    duration: Io.Duration,
    bytes_read: usize,

    fn bytesPerNanoseconds(self: BenchRes) f32 {
        const bytes: f32 = @floatFromInt(self.bytes_read);
        const ns: f32 = @floatFromInt(self.duration.toNanoseconds());
        return bytes / ns;
    }

    pub fn format(
        self: BenchRes,
        writer: *Io.Writer,
    ) !void {
        try writer.print("Read {} bytes in {f}. ({} bytes/ns)", .{
            self.bytes_read,
            self.duration,
            self.bytesPerNanoseconds(),
        });
    }

    pub fn ratio(self: BenchRes, other: BenchRes) f32 {
        const self_ns: f32 = @floatFromInt(self.duration.toNanoseconds());
        const other_ns: f32 = @floatFromInt(other.duration.toNanoseconds());
        return 100.0 * (self_ns / other_ns - 1);
    }
};

fn getAllNVFP4Jobs(
    a: std.mem.Allocator,
    r: *const st.Registry,
) ![]job.Job {
    var jobs = std.ArrayList(job.Job).empty;
    errdefer {
        for (jobs.items) |j| {
            switch (j.dest) {
                .mem => |buffer| a.free(buffer),
                else => {},
            }
        }
        jobs.deinit(a);
    }

    const tensors = try nvfp4.getAllNVFP4Tensors(a, r);
    defer a.free(tensors);

    for (tensors) |t| {
        try jobs.append(a, .{ .dest = .discard, .tensor = .{ .nvfp4 = t } });
    }

    return try jobs.toOwnedSlice(a);
}

fn freeJobs(a: std.mem.Allocator, jobs: []job.Job) void {
    for (jobs) |j| {
        switch (j.dest) {
            .mem => |buffer| {
                a.free(buffer);
            },
            else => {},
        }
    }
    a.free(jobs);
}

fn benchmark(
    a: std.mem.Allocator,
    io: Io,
    filename: []const u8,
    config: nvfp4.DequantConfig,
) !BenchRes {
    const buff_size = 32 * 1024;
    var buffer: [buff_size]u8 = undefined;
    var aw: Io.Writer.Discarding = .init(&buffer);

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    const tensors = try nvfp4.getAllNVFP4Tensors(a, &reg);
    defer a.free(tensors);

    const clock = Clock.awake;
    const start = clock.now(io);

    var packed_buffer: [buff_size]u8 = undefined;
    var scales_buffer: [buff_size]u8 = undefined;

    var bytes: usize = 0;

    for (tensors) |t| {
        var reader = try t.reader(io, .{
            .packed_bytes = &packed_buffer,
            .scales = &scales_buffer,
        }, config);
        defer reader.deinit();

        bytes += try reader.interface.streamRemaining(&aw.writer);
    }

    const end = clock.now(io);
    return .{
        .duration = start.durationTo(end),
        .bytes_read = bytes,
    };
}

fn benchmarkJobs(
    io: Io,
    jobs: []job.Job,
    config: nvfp4.DequantConfig,
) !BenchRes {
    var bytes: usize = 0;
    for (jobs) |*j| {
        j.err = null;
        bytes += j.tensor.byteLen();
    }

    const clock = Clock.awake;
    const start = clock.now(io);
    try job.runAllJobs(io, config, jobs);
    const end = clock.now(io);

    for (jobs) |j| {
        if (j.err) |err| return err;
    }

    return .{
        .duration = start.durationTo(end),
        .bytes_read = bytes,
    };
}

fn benchmarkJobsSequentially(
    io: Io,
    jobs: []job.Job,
    config: nvfp4.DequantConfig,
) !BenchRes {
    var bytes: usize = 0;
    for (jobs) |*j| {
        j.err = null;
        bytes += j.tensor.byteLen();
    }

    const clock = Clock.awake;
    const start = clock.now(io);
    for (jobs) |*j| {
        // Doesn't set err
        try j.runImpl(io, config);
    }
    const end = clock.now(io);

    return .{
        .duration = start.durationTo(end),
        .bytes_read = bytes,
    };
}

const KernelInput = struct {
    packed_bytes: []u8,
    scales: []u8,
    inv_global: f32,
};

fn loadTensorBytes(
    a: std.mem.Allocator,
    io: Io,
    tensor: *const st.Tensor,
) ![]u8 {
    var file_buffer: [32 * 1024]u8 = undefined;
    var reader: st.TensorReader = try .init(io, tensor, .{ .file = &file_buffer });
    defer reader.deinit();

    const bytes = try a.alloc(u8, @intCast(tensor.byte_len));
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

fn loadKernelInputs(
    a: std.mem.Allocator,
    io: Io,
    filename: []const u8,
) ![]KernelInput {
    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    var inputs = std.ArrayList(KernelInput).empty;
    var it = reg.iterator();
    while (it.next()) |t| {
        if (std.mem.endsWith(u8, t.key_ptr.*, ".weight_packed")) {
            const base = std.mem.cutSuffix(u8, t.key_ptr.*, ".weight_packed") orelse unreachable;
            const val = try nvfp4.Tensor.fromRegistry(&reg, base) orelse
                return error.MissingTensor;
            try inputs.append(a, .{
                .packed_bytes = try loadTensorBytes(a, io, val.weight_packed),
                .scales = try loadTensorBytes(a, io, val.weight_scale),
                .inv_global = 1.0 / try val.weight_global_scale.readScalar(io, f32),
            });
        }
    }
    return inputs.toOwnedSlice(a);
}

fn kernelBenchmark(
    io: Io,
    inputs: []const KernelInput,
    config: nvfp4.DequantConfig,
    out: []f32,
) !BenchRes {
    const chunk_blocks = out.len / 16;

    const clock = Clock.awake;
    const start = clock.now(io);

    var bytes: usize = 0;

    for (inputs) |input| {
        var i: usize = 0;
        while (i < input.scales.len) : (i += chunk_blocks) {
            const k = @min(chunk_blocks, input.scales.len - i);
            try nvfp4.dequantRange(
                config,
                input.packed_bytes[i * 8 ..][0 .. k * 8],
                input.scales[i..][0..k],
                input.inv_global,
                out[0 .. k * 16],
            );
            std.mem.doNotOptimizeAway(out.ptr);
            bytes += k * 16 * @sizeOf(f32);
        }
    }

    const end = clock.now(io);
    return .{
        .duration = start.durationTo(end),
        .bytes_read = bytes,
    };
}

test "all benchmarks work with all configs" {
    const builtin = @import("builtin");
    // Don't run in debug because it takes forever
    if (builtin.mode == .debug) return;
    const a = std.heap.c_allocator;
    var threaded: Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const filename = "nvfp4/model.safetensors";

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();
    const jobs = try getAllNVFP4Jobs(a, &reg);
    defer freeJobs(a, jobs);
    for ([_]nvfp4.DequantAlgo{ .sequential, .simd }) |ord| {
        for ([_]nvfp4.e2m1Alg{ .bits, .look_up_table, .tbl }) |alg| {
            const config: nvfp4.DequantConfig = .{ .ordering = ord, .e2m1 = alg };
            _ = try benchmark(a, io, filename, config);
            _ = try benchmarkJobs(io, jobs, config);
            _ = try benchmarkJobsSequentially(io, jobs, config);
        }
    }
}

fn runBenchmark(
    self: Benchmark,
    config: nvfp4.DequantConfig,
    a: std.mem.Allocator,
    io: Io,
    filename: []const u8,
) !BenchRes {
    const res = try self.run(config, a, io, filename);
    const name = self.name();
    print("{s} benchmark: {f}\n", .{ name, res });
    return res;
}

const Benchmark = union(enum) {
    kernel: struct {
        out: []f32,
        is_big: bool,
    },
    regular,
    jobs,
    jobs_seq,

    fn name(self: Benchmark) []const u8 {
        switch (self) {
            .kernel => |info| return if (info.is_big)
                "Kernel (BIG)"
            else
                "Kernel (SMALL)",
            .regular => return "Regular",
            .jobs => return "Jobs",
            .jobs_seq => return "Jobs SEQ",
        }
    }

    fn run(
        self: Benchmark,
        config: nvfp4.DequantConfig,
        a: std.mem.Allocator,
        io: Io,
        filename: []const u8,
    ) !BenchRes {
        switch (self) {
            .kernel => |info| {
                var arena: std.heap.ArenaAllocator = .init(a);
                defer arena.deinit();
                const inputs = try loadKernelInputs(arena.allocator(), io, filename);
                return try kernelBenchmark(io, inputs, config, info.out);
            },
            .regular => return try benchmark(a, io, filename, config),
            .jobs => {
                var reg: st.Registry = try .fromPath(a, io, filename);
                defer reg.deinit();
                const jobs = try getAllNVFP4Jobs(a, &reg);
                defer freeJobs(a, jobs);
                return try benchmarkJobs(io, jobs, config);
            },
            .jobs_seq => {
                var reg: st.Registry = try .fromPath(a, io, filename);
                defer reg.deinit();
                const jobs = try getAllNVFP4Jobs(a, &reg);
                defer freeJobs(a, jobs);
                return try benchmarkJobsSequentially(io, jobs, config);
            },
        }
    }
};

pub fn main() !void {
    const a = std.heap.c_allocator;
    var threaded: Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const filename = "nvfp4/model.safetensors";

    const small_out = try a.alloc(f32, 16 * 1024 / @sizeOf(f32));
    defer a.free(small_out);

    const big_out = try a.alloc(f32, 64 * 1024 * 1024 / @sizeOf(f32));
    defer a.free(big_out);

    const benchmarks = [_]Benchmark{
        .jobs,
        .jobs_seq,
        .{ .kernel = .{ .out = big_out, .is_big = true } },
        .{ .kernel = .{ .out = small_out, .is_big = false } },
        .regular,
    };
    const ords = [_]nvfp4.DequantAlgo{ .sequential, .simd };
    const algs = [_]nvfp4.e2m1Alg{ .bits, .look_up_table, .tbl };

    const warmup = 3;

    for (0..warmup) |_| {
        for (benchmarks) |b| {
            for (ords) |ord| {
                for (algs) |alg| {
                    _ = try b.run(.{ .ordering = ord, .e2m1 = alg }, a, io, filename);
                }
            }
        }
    }

    var min_b: Benchmark = undefined;
    var min_config: nvfp4.DequantConfig = undefined;
    var min_res: BenchRes = undefined;

    var first = true;
    for (benchmarks) |b| {
        for (ords) |ord| {
            for (algs) |alg| {
                const config: nvfp4.DequantConfig = .{ .ordering = ord, .e2m1 = alg };
                print("Config = {}\n", .{config});
                const res = try runBenchmark(b, config, a, io, filename);
                if (first or res.duration.toMilliseconds() < min_res.duration.toMilliseconds()) {
                    min_b = b;
                    min_config = config;
                    min_res = res;
                }
                print("\n", .{});
                if (first) first = false;
            }
        }
    }

    print("----------------------------------\n", .{});
    print("Best results:\n", .{});
    print("Config = {}\n", .{min_config});
    print("{s} benchmark: {f}\n", .{ min_b.name(), min_res });
}
