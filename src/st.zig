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
    pub fn into_tensorInfo(self: @This()) TensorInfo {
        const offset = self.info.data_offsets[0];
        const byte_len = self.info.data_offsets[1] - offset;
        return .{
            .name = self.name,
            .dtype = self.info.dtype,
            .shape = self.info.shape,
            .byte_len = byte_len,
            .offset = offset,
        };
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DataType,
    shape: []u64,
    offset: usize,
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
        errdefer arena.deinit();
        const a = arena.allocator();

        const root = try std.json.parseFromSliceLeaky(
            std.json.Value,
            a,
            blob,
            .{ .allocate = .alloc_always },
        );

        var tensors: std.ArrayList(TensorInfo) = .empty;
        var it = root.object.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
            const raw: RawTensor = .{
                .name = entry.key_ptr.*,
                .info = try std.json.parseFromValueLeaky(RawTensorInfo, a, entry.value_ptr.*, .{}),
            };
            const info = raw.into_tensorInfo();
            try tensors.append(a, info);
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

    n: usize,
    header: SafeTensorHeader,
    io: std.Io,
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
        return .{ .header = header, .io = io, .file = file, .n = n };
    }

    pub fn close(self: *Self) void {
        self.file.close(self.io);
        self.header.deinit();
    }

    pub fn count(self: *Self) usize {
        return self.header.tensors.len;
    }

    pub fn header_offset_bytes(self: *const Self) usize {
        return self.n + 8;
    }

    pub fn iterator(self: *const Self) Iterator {
        return .{
            .index = 0,
            .f = self,
        };
    }

    pub fn readBytes(self: *const Self, a: std.mem.Allocator, info: *const TensorInfo) ![]u8 {
        var read_buf: [READ_BUFFER_CAP]u8 = undefined;
        var reader: std.Io.File.Reader = self.file.reader(self.io, &read_buf);
        try reader.seekTo(self.header_offset_bytes() + info.offset);
        const res = try reader.interface.readAllocAll(a, info.byte_len);
        return res;
    }

    pub const Iterator = struct {
        index: usize,
        f: *const File,

        pub fn next(self: *@This()) ?*TensorInfo {
            if (self.f.header.tensors.len <= self.index) {
                return null;
            }
            const info = &self.f.header.tensors[self.index];
            self.index += 1;
            return info;
        }

        pub fn reset(self: *@This()) void {
            self.index = 0;
        }
    };

    /// View type over a `File` for a specific NVFP4 encoded layer
    pub const NVFP4Layer = struct {
        weight_global_scale: *TensorInfo,
        weight_scale: *TensorInfo,
        weight_packed: *TensorInfo,

        f: *File,
    };

    pub fn getNVFP4Layer(self: *Self, base: []const u8) ?NVFP4Layer {
        var weight_global_scale: ?*TensorInfo = null;
        var weight_scale: ?*TensorInfo = null;
        var weight_packed: ?*TensorInfo = null;

        var it = self.iterator();
        while (it.next()) |t| {
            if (!std.mem.startsWith(u8, t.name, base)) {
                continue;
            }
            if (std.mem.endsWith(u8, t.name, "weight_global_scale")) {
                weight_global_scale = t;
            }
            if (std.mem.endsWith(u8, t.name, "weight_scale")) {
                weight_scale = t;
            }
            if (std.mem.endsWith(u8, t.name, "weight_packed")) {
                weight_packed = t;
            }
        }

        return .{
            .weight_global_scale = weight_global_scale orelse return null,
            .weight_scale = weight_scale orelse return null,
            .weight_packed = weight_packed orelse return null,
            .f = self,
        };
    }
};
