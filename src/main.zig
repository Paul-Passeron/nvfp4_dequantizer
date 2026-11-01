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

fn benchmark(
    a: std.mem.Allocator,
    io: Io,
    filename: []const u8,
    algo: nvfp4.DequantAlgo,
) !BenchRes {
    const buff_size = 32 * 1024;
    var buffer: [buff_size]u8 = undefined;
    var aw: Io.Writer.Discarding = .init(&buffer);

    var reg: st.Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    var tensors = std.ArrayList(nvfp4.Tensor).empty;
    var it = reg.iterator();
    while (it.next()) |t| {
        if (std.mem.endsWith(u8, t.key_ptr.*, ".weight_packed")) {
            const base = std.mem.cutSuffix(u8, t.key_ptr.*, ".weight_packed") orelse unreachable;
            const val = try nvfp4.Tensor.fromRegistry(&reg, base) orelse
                return error.MissingTensor;
            try tensors.append(a, val);
        }
    }
    const clock = Clock.awake;
    const start = clock.now(io);

    var packed_buffer: [buff_size]u8 = undefined;
    var scales_buffer: [buff_size]u8 = undefined;

    var bytes: usize = 0;

    for (tensors.items) |t| {
        var reader = try t.reader(io, .{
            .packed_bytes = &packed_buffer,
            .scales = &scales_buffer,
        }, algo);
        defer reader.deinit();

        bytes += try reader.interface.streamRemaining(&aw.writer);
    }

    const end = clock.now(io);
    return .{
        .duration = start.durationTo(end),
        .bytes_read = bytes,
    };
}

pub fn main() !void {
    const a = std.heap.c_allocator;
    var threaded: Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const filename = "nvfp4/model.safetensors";

    const warmer_count = 4;
    var seq: BenchRes = undefined;
    var simd: BenchRes = undefined;
    for (0..warmer_count) |_| {
        seq = try benchmark(a, io, filename, .sequential);
        simd = try benchmark(a, io, filename, .simd);
    }
    print("Sequential benchmark: {f}\n", .{seq});
    print("SIMD benchmark: {f}\n", .{simd});
    print(
        "SIMD benchmark / Sequential benchmark: {d:.[1]}%\n",
        .{ simd.ratio(seq), 2 },
    );
}
