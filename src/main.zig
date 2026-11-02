const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const Io = std.Io;
const Clock = Io.Clock;
const print = std.debug.print;

const BenchRes = struct {
    duration: Io.Duration,
    bytes_read: usize,

    fn bytes_per_ns(self: BenchRes) f32 {
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
            self.bytes_per_ns(),
        });
    }

    pub fn ratio(self: BenchRes, other: BenchRes) f32 {
        const self_ns: f32 = @floatFromInt(self.duration.toNanoseconds());
        const other_ns: f32 = @floatFromInt(other.duration.toNanoseconds());
        return 100.0 * (self_ns / other_ns - 1);
    }
};

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

fn getAllNVFP4Jobs(
    a: std.mem.Allocator,
    r: *const st.Registry,
) ![]Job {
    var jobs = std.ArrayList(Job).empty;
    errdefer {
        for (jobs.items) |j| {
            switch (j.dest) {
                .mem => |buffer| a.free(buffer),
                else => {},
            }
        }
        jobs.deinit(a);
    }

    const tensors = try getAllNVFP4Tensors(a, r);
    defer a.free(tensors);

    for (tensors) |t| {
        try jobs.append(a, .{ .dest = .discard, .tensor = t });
    }

    return try jobs.toOwnedSlice(a);
}

fn freeJobs(a: std.mem.Allocator, jobs: []Job) void {
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

    const tensors = try getAllNVFP4Tensors(a, &reg);
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
    jobs: []Job,
    config: nvfp4.DequantConfig,
) !BenchRes {
    var bytes: usize = 0;
    for (jobs) |*j| {
        j.err = null;
        bytes += j.tensor.dequantByteLen();
    }

    const clock = Clock.awake;
    const start = clock.now(io);
    try dequantizeAll(io, config, jobs);
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
    jobs: []Job,
    config: nvfp4.DequantConfig,
) !BenchRes {
    var bytes: usize = 0;
    for (jobs) |*j| {
        j.err = null;
        bytes += j.tensor.dequantByteLen();
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

fn runKernelBenchmarks(
    a: std.mem.Allocator,
    io: Io,
    inputs: []const KernelInput,
    name: []const u8,
    out_size: usize,
) !void {
    const out = try a.alloc(f32, out_size / @sizeOf(f32));
    defer a.free(out);

    const warmer_count = 4;
    var seq: BenchRes = undefined;
    var simd: BenchRes = undefined;
    for (0..warmer_count) |_| {
        seq = try kernelBenchmark(io, inputs, .{ .ordering = .sequential }, out);
        simd = try kernelBenchmark(io, inputs, .{ .ordering = .simd }, out);
    }
    print("Sequential kernel ({s} output): {f}\n", .{ name, seq });
    print("SIMD kernel ({s} output): {f}\n", .{ name, simd });
    print(
        "SIMD kernel / Sequential kernel ({s} output): {d:.[2]}%\n",
        .{ name, simd.ratio(seq), 2 },
    );
}

// TODO: Look into how the ZML framework actually does multicore io

pub const Destination = union(enum) {
    /// length must be `tensor.dequantByteLen()`
    mem: []u8,
    discard,
};

pub const Job = struct {
    tensor: nvfp4.Tensor,

    dest: Destination,

    err: ?anyerror = null,

    fn run(self: *Job, io: Io, config: nvfp4.DequantConfig) void {
        self.runImpl(io, config) catch |err| {
            self.err = err;
            return;
        };
    }

    fn runImpl(self: *Job, io: Io, config: nvfp4.DequantConfig) !void {
        var packed_buffer: [32 * 1024]u8 = undefined;
        var scale_buffer: [32 * 1024]u8 = undefined;
        var reader = try self.tensor.reader(io, .{
            .packed_bytes = &packed_buffer,
            .scales = &scale_buffer,
        }, config);
        defer reader.deinit();
        switch (self.dest) {
            .mem => |buffer| {
                var w = Io.Writer.fixed(buffer);
                _ = try reader.interface.streamRemaining(&w);
            },
            .discard => {
                var buffer: [32 * 1024]u8 = undefined;
                var w = Io.Writer.Discarding.init(&buffer);
                _ = try reader.interface.streamRemaining(&w.writer);
            },
        }
    }
};

fn dequantizeAll(io: Io, config: nvfp4.DequantConfig, jobs: []Job) Io.Cancelable!void {
    var group: Io.Group = .init;
    for (jobs) |*job| {
        group.async(io, Job.run, .{ job, io, config });
    }
    try group.await(io);
}

test "dequantizeAll on empty" {
    const a = std.heap.c_allocator;
    var threaded: Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    _ = try dequantizeAll(io, .{ .ordering = .sequential }, &.{});
}

pub fn main() !void {
    const a = std.heap.c_allocator;
    var threaded: Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const filename = "nvfp4/model.safetensors";

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();
    const jobs = try getAllNVFP4Jobs(a, &reg);
    defer freeJobs(a, jobs);

    const warmer_count = 4;
    var seq: BenchRes = undefined;
    var simd: BenchRes = undefined;
    var seq_job: BenchRes = undefined;
    var simd_job: BenchRes = undefined;
    var seq_job_seq: BenchRes = undefined;
    var simd_job_seq: BenchRes = undefined;

    for (0..warmer_count) |_| {
        seq = try benchmark(a, io, filename, .{ .ordering = .sequential });
        simd = try benchmark(a, io, filename, .{ .ordering = .simd });
        seq_job = try benchmarkJobs(io, jobs, .{ .ordering = .sequential });
        simd_job = try benchmarkJobs(io, jobs, .{ .ordering = .simd });
        seq_job_seq = try benchmarkJobsSequentially(io, jobs, .{ .ordering = .sequential });
        simd_job_seq = try benchmarkJobsSequentially(io, jobs, .{ .ordering = .simd });
    }
    print("Sequential benchmark: {f}\n", .{seq});
    print("SIMD benchmark: {f}\n", .{simd});
    print(
        "SIMD benchmark / Sequential benchmark: {d:.[1]}%\n",
        .{ simd.ratio(seq), 2 },
    );
    print("\n", .{});

    print("Sequential JOB benchmark: {f}\n", .{seq_job});
    print("SIMD JOB benchmark: {f}\n", .{simd_job});
    print(
        "SIMD JOB benchmark / Sequential JOB benchmark: {d:.[1]}%\n",
        .{ simd_job.ratio(seq_job), 2 },
    );
    print("\n", .{});

    print("Sequential JOB SEQ benchmark: {f}\n", .{seq_job_seq});
    print("SIMD JOB SEQ benchmark: {f}\n", .{simd_job_seq});
    print(
        "SIMD JOB SEQ benchmark / Sequential JOB SEQ benchmark: {d:.[1]}%\n",
        .{ simd_job_seq.ratio(seq_job_seq), 2 },
    );
    print("\n", .{});
    print(
        "SIMD JOB benchmark / Sequential benchmark: {d:.[1]}%\n",
        .{ simd_job.ratio(seq), 2 },
    );
    print("\n", .{});

    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const inputs = try loadKernelInputs(arena.allocator(), io, filename);

    try runKernelBenchmarks(a, io, inputs, "16KiB", 16 * 1024);
    print("\n", .{});
    try runKernelBenchmarks(a, io, inputs, "64MiB", 64 * 1024 * 1024);
}
