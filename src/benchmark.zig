const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const job = @import("job");
const Io = std.Io;
const Clock = Io.Clock;
const print = std.debug.print;

const all_dequant = [_]nvfp4.DequantAlgo{ .sequential, .simd };
const all_e2m1_alg = [_]nvfp4.e2m1Alg{ .bits, .look_up_table, .tbl };
const all_e4m3_alg = [_]nvfp4.e4m3Alg{ .bits, .look_up_table };

const config_count =
    all_dequant.len *
    all_e2m1_alg.len *
    all_e4m3_alg.len;

const all_configs: [config_count]nvfp4.DequantConfig = blk: {
    var res: [config_count]nvfp4.DequantConfig = undefined;
    var i = 0;
    for (all_dequant) |d| {
        for (all_e2m1_alg) |e2m1| {
            for (all_e4m3_alg) |e4m3| {
                res[i] = .{
                    .dequant = d,
                    .e2m1 = e2m1,
                    .e4m3 = e4m3,
                };
                i += 1;
            }
        }
    }
    break :blk res;
};

const all_datatypes = [_]st.DataType{ .F32, .BF16, .F16 };

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
    dt: st.DataType,
    r: *const st.Registry,
) ![]job.Job {
    const tensors = try nvfp4.getAllNVFP4Tensors(a, dt, r);
    defer a.free(tensors);

    var jobs = try a.alloc(job.Job, tensors.len);
    errdefer freeJobs(a, jobs);

    for (tensors, 0..) |t, i| {
        jobs[i] = .{
            .dest = .discard,
            .tensor = .{ .nvfp4 = t },
        };
    }

    return jobs;
}

fn freeJobs(a: std.mem.Allocator, jobs: []job.Job) void {
    for (jobs) |j| switch (j.dest) {
        .mem => |buffer| a.free(buffer),
        else => {},
    };
    a.free(jobs);
}

fn benchmark(
    a: std.mem.Allocator,
    io: Io,
    dt: st.DataType,
    filename: []const u8,
    config: nvfp4.DequantConfig,
) !BenchRes {
    const buff_size = 32 * 1024;
    var buffer: [buff_size]u8 = undefined;
    var aw: Io.Writer.Discarding = .init(&buffer);

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    const tensors = try nvfp4.getAllNVFP4Tensors(a, dt, &reg);
    defer a.free(tensors);

    const clock = Clock.awake;
    const start = clock.now(io);

    var packed_buffer: [buff_size]u8 = undefined;
    var scales_buffer: [buff_size]u8 = undefined;

    var bytes: usize = 0;

    for (tensors) |t| {
        var reader = try t.reader(a, io, .{
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
    allocator: std.mem.Allocator,
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
    try job.runAllJobs(io, allocator, config, jobs);
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
    allocator: std.mem.Allocator,
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
        try j.runImpl(allocator, io, config);
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
    var reader: st.TensorReader = try .init(io, tensor, .{
        .file = &file_buffer,
    });
    defer reader.deinit();

    const bytes = try a.alloc(u8, @intCast(tensor.byte_len));
    try reader.interface.readSliceAll(bytes);
    return bytes;
}

fn loadKernelInputs(
    a: std.mem.Allocator,
    io: Io,
    dt: st.DataType,
    filename: []const u8,
) ![]KernelInput {
    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    var inputs = std.ArrayList(KernelInput).empty;
    var it = reg.iterator();
    while (it.next()) |t| {
        if (!std.mem.endsWith(u8, t.key_ptr.*, ".weight_packed"))
            continue;
        const base = std.mem.cutSuffix(
            u8,
            t.key_ptr.*,
            ".weight_packed",
        ) orelse unreachable;
        const val = try nvfp4.Tensor.fromRegistry(dt, &reg, base) orelse
            return error.MissingTensor;
        const global = try val.weight_global_scale.readScalar(io, f32);
        const bytes = try loadTensorBytes(a, io, val.weight_packed);
        const scales = try loadTensorBytes(a, io, val.weight_scale);
        try inputs.append(a, .{
            .packed_bytes = bytes,
            .scales = scales,
            .inv_global = 1.0 / global,
        });
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
                f32,
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
    for (all_datatypes) |dt| {
        const jobs = try getAllNVFP4Jobs(a, dt, &reg);
        defer freeJobs(a, jobs);
        for (all_configs) |config| {
            _ = try benchmark(a, io, dt, filename, config);
            _ = try benchmarkJobs(io, a, jobs, config);
            _ = try benchmarkJobsSequentially(io, a, jobs, config);
        }
    }
}

fn runBenchmark(
    self: Benchmark,
    config: nvfp4.DequantConfig,
    a: std.mem.Allocator,
    io: Io,
    dt: st.DataType,
    filename: []const u8,
) !BenchRes {
    const res = try self.run(config, a, io, dt, filename);
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
        dt: st.DataType,
        filename: []const u8,
    ) !BenchRes {
        switch (self) {
            .kernel => |info| {
                var arena: std.heap.ArenaAllocator = .init(a);
                defer arena.deinit();
                const inputs = try loadKernelInputs(
                    arena.allocator(),
                    io,
                    dt,
                    filename,
                );
                return try kernelBenchmark(io, inputs, config, info.out);
            },
            .regular => return try benchmark(a, io, dt, filename, config),
            .jobs => {
                var reg: st.Registry = try .fromPath(a, io, filename);
                defer reg.deinit();
                const jobs = try getAllNVFP4Jobs(a, dt, &reg);
                defer freeJobs(a, jobs);
                return try benchmarkJobs(io, a, jobs, config);
            },
            .jobs_seq => {
                var reg: st.Registry = try .fromPath(a, io, filename);
                defer reg.deinit();
                const jobs = try getAllNVFP4Jobs(a, dt, &reg);
                defer freeJobs(a, jobs);
                return try benchmarkJobsSequentially(io, a, jobs, config);
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

    const all_benchmarks = [_]Benchmark{
        .jobs,
        .jobs_seq,
        .{ .kernel = .{ .out = big_out, .is_big = true } },
        .{ .kernel = .{ .out = small_out, .is_big = false } },
        .regular,
    };

    const warmup = 1;

    for (0..warmup) |_| {
        for (all_datatypes) |dt| {
            for (all_benchmarks) |b| {
                for (all_configs) |config| {
                    _ = try b.run(config, a, io, dt, filename);
                }
            }
        }
    }

    var min_b: Benchmark = undefined;
    var min_config: nvfp4.DequantConfig = undefined;
    var min_res: BenchRes = undefined;
    var min_dt: st.DataType = undefined;

    var first = true;
    for (all_benchmarks) |b| {
        for (all_datatypes) |dt| {
            for (all_configs) |config| {
                print("Config = {}\n", .{config});
                const res = try runBenchmark(b, config, a, io, dt, filename);
                const r = res.duration.toMilliseconds();
                const rm = min_res.duration.toMilliseconds();
                if (first or r < rm) {
                    min_b = b;
                    min_config = config;
                    min_res = res;
                    min_dt = dt;
                }
                print("\n", .{});
                if (first) first = false;
            }
        }
    }

    print("----------------------------------\n", .{});
    print("Best results:\n", .{});
    print("Datatype = {}\n", .{min_dt});
    print("Config = {}\n", .{min_config});
    print("{s} benchmark: {f}\n", .{ min_b.name(), min_res });
}
