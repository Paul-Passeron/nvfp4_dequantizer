const std = @import("std");

pub const DataType = enum { U8, U16, U32, U64, F8, F16, F32, F64, BF16, F8_E4M3 };

pub const TensorInfo = struct {
    dtype: DataType,
    shape: []u64,
    data_offsets: [2]usize,

    const Self = @This();
};

pub const Tensor = struct {
    name: []const u8,
    info: TensorInfo,
};

pub const Metadata = struct {};

pub const SafeTensorHeader = struct {
    arena: std.heap.ArenaAllocator,

    tensors: []Tensor,
    metadata: Metadata,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    pub fn fromJSON(allocator: std.mem.Allocator, blob: []const u8) !Self {
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

pub const File = struct {
    const Self = @This();
    pub fn open(
        io: std.Io,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) !Self {
        _ = io; // autofix
        _ = allocator; // autofix
        _ = path; // autofix

        @panic("TODO");
    }
};
