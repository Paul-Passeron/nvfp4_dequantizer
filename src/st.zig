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
        var file = try std.Io.Dir.cwd().openFile(io, path, .{
            .mode = .read_only,
        });

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

    pub fn readScalar(self: *const Self, comptime T: type, info: *const TensorInfo) !T {
        // TODO: not sure this is even REMOTELY correct ^^'
        var read_buf: [@sizeOf(T)]u8 = undefined;
        var bytes: [@sizeOf(T)]u8 = undefined;

        var reader: std.Io.File.Reader = self.file.reader(self.io, &read_buf);
        try reader.seekTo(self.header_offset_bytes() + info.offset);

        try reader.interface.readSliceAll(&bytes);

        return @bitCast(std.mem.readInt(
            @Int(.unsigned, @bitSizeOf(T)),
            &bytes,
            .little,
        ));
    }

    pub const Iterator = struct {
        index: usize,
        f: *const File,

        pub fn next(self: *@This()) ?*const TensorInfo {
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

    fn expectedDataType(comptime T: type) ?DataType {
        switch (T) {
            f16 => return .F16,
            f32 => return .F32,
            f64 => return .F64,
            u8 => return .U8,
            u16 => return .U16,
            u32 => return .U32,
            u64 => return .U64,
            else => return null,
        }
    }

    pub fn tensorFromInfo(
        self: *const Self,
        comptime T: type,
        infos: *const TensorInfo,
    ) !?BasicTensor(T) {
        return .{
            .weights = infos,
            .file = self,
        };
    }

    pub fn tensor(
        self: *const Self,
        comptime T: type,
        name: []const u8,
    ) error{WrongDataType}!?BasicTensor(T) {
        var infos: ?*const TensorInfo = null;
        var it = self.iterator();
        while (it.next()) |info| {
            if (!std.mem.eql(u8, info.name, name)) {
                continue;
            }
            if (info.dtype != expectedDataType(T)) {
                return error.WrongDataType;
            }
            infos = info;
            break;
        }
        const i = infos orelse return null;
        return self.tensorFromInfo(T, i);
    }

    pub fn BasicTensor(comptime T: type) type {
        return struct {
            weights: *const TensorInfo,
            file: *const File,

            const ThisRawTensor = @This();

            pub fn reader(
                self: *const ThisRawTensor,
                a: std.mem.Allocator,
            ) !BlockReader {
                const buffer = try a.alloc(u8, READ_BUFFER_CAP);
                errdefer a.free(buffer);
                var r = self.file.file.reader(self.file.io, buffer);
                try r.seekTo(self.file.header_offset_bytes() + self.weights.offset);
                return .{
                    .r = r,
                    .buffer = buffer,
                    .a = a,
                    .tensor = self,
                    .i = 0,
                };
            }

            pub const BlockReader = struct {
                r: std.Io.File.Reader,
                buffer: []u8,
                a: std.mem.Allocator,
                tensor: *const ThisRawTensor,
                i: usize,

                pub fn deinit(self: *BlockReader) void {
                    self.a.free(self.buffer);
                }

                pub fn next(self: *BlockReader) !?T {
                    if (self.i >= self.tensor.weights.byte_len) {
                        return null;
                    }
                    var res: [1]T = undefined;
                    try self.r.interface.readSliceAll(@ptrCast(&res));
                    self.i += 1;
                    return res[0];
                }
            };
        };
    }
};
