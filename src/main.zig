const std = @import("std");

// Found the original NVIDIA paper at: <https://arxiv.org/pdf/2509.25149>

// From <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html>:
// The representation of an NVFP4 tensor element x is given by:
// `x = x_e2m1 * s_block * s_global`
// where
//
// `x_e2m1` is the 4-bit value,
// `s_block` is a local FP8 E4M3 scaling factor shared by a block of 16 consecutive elements,
// `s_global` is a global FP32 scaling factor applied to the entire tensor.

// Representations:
// - NVFP4:
//      1 sign bit
//      2 exponent bits
//      1 mantissa bit
//
// - FP8 E4M3:
//      1 sign bit
//      4 exponent bits
//      3 mantissa bits

const DataType = enum { U8, U16, U32, U64, F8, F16, F32, F64, BF16, F8_E4M3 };

const TensorInfo = struct {
    dtype: DataType,
    shape: []u64,
    data_offsets: [2]usize,

    const Self = @This();
};

const Tensor = struct {
    name: []const u8,
    info: TensorInfo,
};

const Metadata = struct {};

const SafeTensorHeader = struct {
    arena: std.heap.ArenaAllocator,

    tensors: []Tensor,
    metadata: Metadata,

    const Self = @This();

    fn deinit(self: *const Self) void {
        self.arena.deinit();
    }

    fn fromJSON(allocator: std.mem.Allocator, blob: []const u8) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        const a = arena.allocator();

        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, blob, .{
            .allocate = .alloc_always,
        });

        var tensors: std.ArrayList(Tensor) = .empty;
        var it = root.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            try tensors.append(a, .{
                .name = entry.key_ptr.*,
                .info = try std.json.parseFromValueLeaky(TensorInfo, a, entry.value_ptr.*, .{}),
            });
        }

        return .{
            .tensors = try tensors.toOwnedSlice(a),
            .metadata = .{},
            .arena = arena,
        };
    }
};

const FOUR_KB = 4 * 1024;

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var write_buf: [FOUR_KB]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &write_buf);
    const w = &fw.interface;

    // const filename = "converted/model-00001-of-00001.safetensors";
    const filename = "nvfp4/model.safetensors";
    var file = try std.Io.Dir.cwd().openFile(io, filename, .{ .mode = .read_only });
    defer file.close(io);
    var read_buf: [FOUR_KB]u8 = undefined;
    var f_reader: std.Io.File.Reader = file.reader(io, &read_buf);

    const n = try f_reader.interface.takeInt(u64, .little);
    try w.print("N is {}\n", .{n});

    const json_blob = try f_reader.interface.readAlloc(allocator, n);
    defer allocator.free(json_blob);

    const header = try SafeTensorHeader.fromJSON(allocator, json_blob);
    defer header.deinit();

    const M = std.AutoHashMap(DataType, void);
    var m: M = .init(allocator);
    defer m.deinit();

    for (0.., header.tensors) |i, t| {
        if (t.info.dtype != .U8) {
            continue;
        }
        try w.print("u8 at {}\n", .{i});
        try f_reader.seekTo(t.info.data_offsets[0]);
        const len = t.info.data_offsets[1] - t.info.data_offsets[0];
        const r = &f_reader.interface;
        for (0..len) |_| {
            const b = try r.takeByte();
            const lo: u4 = @intCast(b & 0b1111);
            const hi: u4 = @intCast(b >> 4 & 0b1111);
            try w.print("{}: {} and {}\n", .{ b, lo, hi });
        }
        break;
    }

    try w.flush();
}
