const std = @import("std");

pub const DataType = enum { U8, U16, U32, U64, F8, F16, F32, F64, BF16, F8_E4M3 };

pub const RawTensorInfo = struct {
    dtype: DataType,
    shape: []u64,
    data_offsets: [2]usize,

    const Self = @This();
};

pub const RawTensor = struct {
    name: []const u8,
    info: RawTensorInfo,

    /// Takes ownership of the values inside the tensor.
    pub fn into_tensorInfo(self: @This()) struct { info: TensorInfo, offset: usize } {
        const offset = self.info.data_offsets[0];
        const byte_len = self.info.data_offsets[1] - offset;
        const res = .{
            .name = self.name,
            .dtype = self.dtype,
            .shape = self.shape,
            .byte_len = byte_len,
        };
        return res;
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DataType,
    shape: []u64,
    byte_len: usize,
};

pub const Metadata = struct {};

pub const SafeTensorHeader = struct {
    arena: std.heap.ArenaAllocator,

    tensors: []TensorInfo,
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

        var tensors: std.ArrayList(RawTensor) = .empty;
        var it = root.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            try tensors.append(a, .{
                .name = entry.key_ptr.*,
                .info = try std.json.parseFromValueLeaky(RawTensorInfo, a, entry.value_ptr.*, .{}),
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

    header: SafeTensorHeader,
    reader: std.Io.File.Reader,
    file: std.Io.File,

    const READ_BUFFER_CAP = 4096;

    pub fn open(
        io: std.Io,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) !Self {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
        var read_buf: [READ_BUFFER_CAP]u8 = undefined;
        var reader: std.Io.File.Reader = file.reader(io, &read_buf);
        const n = try reader.interface.takeInt(u64, .little);
        const json_blob = try reader.interface.readAlloc(allocator, n);
        defer allocator.free(json_blob);
        const header = try SafeTensorHeader.fromJSON(allocator, json_blob);
        return .{ header, reader, file };
    }

    pub fn close(self: *Self) void {
        self.file.close();
        self.header.deinit();
    }

    pub fn count(self: *Self) usize {
        return self.header.tensors.len;
    }

    const TensorInfoIterator = struct {};
};
