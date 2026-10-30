const std = @import("std");

pub const DataType = enum { U8, U16, U32, U64, F16, F32, F64, BF16, F8_E4M3 };

pub const RawTensorInfo = struct {
    dtype: DataType,
    shape: []u64,
    data_offsets: [2]usize,
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

    pub fn headerOffsetBytes(self: *const Self) usize {
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
        try reader.seekTo(self.headerOffsetBytes() + info.offset);

        const res = try reader.interface.readAllocAll(a, info.byte_len);

        return res;
    }

    pub fn readScalar(self: *const Self, comptime T: type, info: *const TensorInfo) !T {
        // TODO: not sure this is even REMOTELY correct ^^'
        var read_buf: [@sizeOf(T)]u8 = undefined;
        var bytes: [@sizeOf(T)]u8 = undefined;

        var reader: std.Io.File.Reader = self.file.reader(self.io, &read_buf);
        try reader.seekTo(self.headerOffsetBytes() + info.offset);

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
                try r.seekTo(self.file.headerOffsetBytes() + self.weights.offset);
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
                    self.i += @sizeOf(T);
                    return res[0];
                }
            };
        };
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

// New API (test)

// Have a map from name to tensors
pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

// Mostly the same as TensorInfo but has its path
pub const Tensor = struct {
    name: []const u8,
    // file path of the safetensor shard that contains it
    path: []const u8,

    dtype: DataType,
    shape: []const u64,

    // offset directly in the file, not from header so
    // we don't store the header length etc...
    offset: u64,
    byte_len: u64,

    pub fn readScalarUnchecked(
        self: *const Tensor,
        io: std.Io,
        comptime T: type,
    ) Errors!T {
        var file = try std.Io.Dir.cwd().openFile(io, self.path, .{
            .mode = .read_only,
        });
        defer file.close(io);

        var buffer: [@sizeOf(T)]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try reader.seekTo(self.offset);
        const r = &reader.interface;
        const slice = try r.take(@sizeOf(T));
        if (slice.len != @sizeOf(T)) {
            return error.ReadFailed;
        }
        const reinterpreted: []T = @alignCast(std.mem.bytesAsSlice(T, slice));
        return reinterpreted[0];
    }

    const Errors = std.Io.File.OpenError ||
        std.Io.File.Reader.Error ||
        std.Io.File.Reader.SeekError ||
        error{ WrongDataType, WrongLength };

    pub fn readScalar(
        self: *const Tensor,
        io: std.Io,
        comptime T: type,
    ) Errors!T {
        if (self.dtype != expectedDataType(T)) {
            return error.WrongDataType;
        }
        if (self.byte_len != @sizeOf(T)) {
            return error.WrongLength;
        }

        return self.readScalarUnchecked(io, T);
    }
};

test "readScalar works" {
    const allocator = std.heap.c_allocator;
    const filename = "nvfp4/model.safetensors";
    const tensor = "model.layers.0.self_attn.q_proj.weight_global_scale";
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try File.open(io, allocator, filename);
    defer f.close();

    var it = f.iterator();
    var found_info: ?*const TensorInfo = null;
    while (it.next()) |info| {
        if (std.mem.eql(u8, info.name, tensor)) {
            found_info = info;
        }
    }
    const info = found_info orelse return error.DidntFindTensor;

    var t: Tensor = .{
        .name = tensor,
        .path = filename,
        .dtype = info.dtype,
        .shape = info.shape,
        .offset = info.offset + f.headerOffsetBytes(),
        .byte_len = info.byte_len,
    };
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, u8));
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, u16));
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, u32));
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, u64));
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, f16));
    try std.testing.expectError(error.WrongDataType, t.readScalar(io, f64));
    _ = try t.readScalar(io, f32);
}

pub const TensorReader = struct {
    io: std.Io,
    file: std.Io.File,
    file_reader: std.Io.File.Reader,
    remaining: u64,
    interface: std.Io.Reader,

    fn stream(
        io_reader: *std.Io.Reader,
        io_writer: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        const r: *TensorReader = @alignCast(
            @fieldParentPtr("interface", io_reader),
        );
        return r.stream_impl(io_writer, limit);
    }

    fn stream_impl(
        self: *TensorReader,
        io_writer: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        if (self.remaining == 0) {
            return std.Io.Reader.StreamError.EndOfStream;
        }
        const actual_limit = limit.min(.limited(self.remaining));
        const read = try self
            .file_reader
            .interface
            .stream(io_writer, actual_limit);
        self.advance_by(read);
        return read;
    }

    fn advance_by(self: *TensorReader, byte_count: usize) void {
        if (byte_count > self.remaining) {
            // TODO: This shouldn't be allowed ?
            self.remaining = 0;
        } else {
            self.remaining -= byte_count;
        }
    }

    fn discard(
        io_reader: *std.Io.Reader,
        limit: std.Io.Limit,
    ) error{ EndOfStream, ReadFailed }!usize {
        const r: *TensorReader = @alignCast(
            @fieldParentPtr("interface", io_reader),
        );
        return r.discard_impl(limit);
    }

    fn discard_impl(
        self: *TensorReader,
        limit: std.Io.Limit,
    ) error{ EndOfStream, ReadFailed }!usize {
        if (self.remaining == 0) {
            return error.EndOfStream;
        }
        const actual_limit = limit.min(.limited(self.remaining));
        const discarded = try self
            .file_reader
            .interface
            .discard(actual_limit);
        self.advance_by(discarded);
        return discarded;
    }

    fn initInterface(buffer: ?[]u8) std.Io.Reader {
        return .{
            .vtable = &.{
                .stream = stream,
                .discard = discard,
            },
            .buffer = buffer orelse &.{},
            .seek = 0,
            .end = 0,
        };
    }

    pub fn init(
        io: std.Io,
        tensor: *const Tensor,
        buffer: []u8,
    ) !TensorReader {
        var own_buffer: ?[]u8 = null;
        var file_buffer = buffer;
        if (buffer.len > 128) {
            own_buffer = buffer[0..128];
            file_buffer = buffer[128..];
        }

        var file = try std.Io.Dir.cwd().openFile(io, tensor.path, .{
            .mode = .read_only,
        });
        errdefer file.close(io);

        var file_reader = file.reader(io, file_buffer);
        try file_reader.seekTo(tensor.offset);
        const interface = initInterface(own_buffer);

        return .{
            .io = io,
            .file = file,
            .file_reader = file_reader,
            .remaining = tensor.byte_len,
            .interface = interface,
        };
    }

    pub fn deinit(self: *TensorReader) void {
        self.file.close(self.io);
    }
};

test "try tensor reader" {

    // FIXME: Don't use File to open the tensor
    const allocator = std.heap.c_allocator;
    const filename = "nvfp4/model.safetensors";
    const tensor = "model.layers.0.self_attn.q_proj.weight_scale";
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try File.open(io, allocator, filename);
    defer f.close();

    var it = f.iterator();
    var found_info: ?*const TensorInfo = null;
    while (it.next()) |info| {
        if (std.mem.eql(u8, info.name, tensor)) {
            found_info = info;
        }
    }
    const info = found_info orelse return error.DidntFindTensor;

    var t: Tensor = .{
        .name = tensor,
        .path = filename,
        .dtype = info.dtype,
        .shape = info.shape,
        .offset = info.offset + f.headerOffsetBytes(),
        .byte_len = info.byte_len,
    };

    const ts = std.testing;

    var buffer: [4096]u8 = undefined;
    var reader: TensorReader = try .init(io, &t, &buffer);
    defer reader.deinit();
    var r = &reader.interface;
    const remaining = try r.allocRemaining(allocator, .unlimited);
    try ts.expectEqual(remaining.len, info.byte_len);
}

// Trying to do it kind of like the ZML framework
// So a registry that holds the tensors with a map
// Then I'll be able to change the tests to directly
// query the `*Tensor`s, get readers, etc...

pub const Registry = struct {
    // Everything in the registry just lives as long
    // as the registry does.
    arena: std.heap.ArenaAllocator,
    tensors: Tensors,

    pub fn init(
        allocator: std.mem.Allocator,
    ) !Registry {
        return .{
            .arena = .init(allocator),
            .tensors = .empty,
        };
    }

    pub fn fromPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !Registry {
        var this: Registry = try .init(allocator);
        if (std.mem.endsWith(u8, path, ".safetensors")) {
            try this.parseFile(io, path);
        } else if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
            @panic("TODO: handle index files");
        }
        return this;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
    }

    pub fn get(
        self: *const Registry,
        name: []const u8,
    ) ?*const Tensor {
        return self.tensors.getPtr(name);
    }

    pub fn count(self: *const Registry) usize {
        return self.tensors.count();
    }

    pub fn iterator(self: *const Registry) Tensors.Iterator {
        return self.tensors.iterator();
    }

    fn parseFile(
        self: *Registry,
        io: std.Io,
        path: []const u8,
    ) !void {
        const a = self.arena.allocator();
        var file = try std.Io.Dir.cwd().openFile(io, path, .{
            .mode = .read_only,
        });
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader: std.Io.File.Reader = file.reader(io, &read_buf);
        const n = try reader.interface.takeInt(u64, .little);
        const json_blob = try reader.interface.readAlloc(a, n);
        // TODO: this uses the arena allocator so freeing it might not
        // do anything... Fine for now I guess but should be aware of
        // that I think. Either we should keep the child allocator in
        // the registry for that exact reason or we should do something
        // else...
        defer a.free(json_blob);

        const root = try std.json.parseFromSliceLeaky(
            std.json.Value,
            a,
            json_blob,
            .{ .allocate = .alloc_always },
        );
        var it = root.object.iterator();
        while (it.next()) |entry| {
            // TODO: What to do with metadata ?
            // - Keep it as a raw JSON object ?
            // - Do something else ?
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

            // TODO: Have RawTensorInfo be types local to this function,
            // or at least to the registry
            const raw_info = try std.json.parseFromValueLeaky(
                RawTensorInfo,
                a,
                entry.value_ptr.*,
                .{},
            );

            const t: Tensor = .{
                .name = entry.key_ptr.*,
                .path = try self.arena.allocator().dupe(u8, path),
                .shape = raw_info.shape,
                .dtype = raw_info.dtype,
                .offset = raw_info.data_offsets[0] + 8 + n,
                .byte_len = raw_info.data_offsets[1] - raw_info.data_offsets[0],
            };
            try self.tensors.put(a, t.name, t);
        }
    }
};

test "try init registry from a path" {
    const a = std.testing.allocator;
    const filename = "nvfp4/model.safetensors";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var r: Registry = try .fromPath(a, io, filename);
    try std.testing.expect(r.count() > 0);
    defer r.deinit();
}

test "can get a reader out of all tensors" {
    const a = std.testing.allocator;
    const filename = "nvfp4/model.safetensors";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var reg: Registry = try .fromPath(a, io, filename);
    defer reg.deinit();
    try std.testing.expect(reg.count() > 0);
    var it = reg.iterator();
    var buffer: [4096]u8 = undefined;
    while (it.next()) |t| {
        var r: TensorReader =
            try .init(io, t.value_ptr, &buffer);
        defer r.deinit();
    }
}

test "can read out of all tensors" {
    const a = std.testing.allocator;
    const filename = "nvfp4/model.safetensors";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var reg: Registry = try .fromPath(a, io, filename);
    defer reg.deinit();
    try std.testing.expect(reg.count() > 0);
    var it = reg.iterator();
    var buffer: [4096]u8 = undefined;
    var w: std.Io.Writer.Discarding = .init(&.{});
    while (it.next()) |t| {
        var r: TensorReader =
            try .init(io, t.value_ptr, &buffer);
        defer r.deinit();
        const n = try r.interface.streamRemaining(&w.writer);
        try std.testing.expectEqual(t.value_ptr.byte_len, n);
    }
}

// TODO: this test is very slow because we read values one by one!
test "both ways to read agree" {
    const a = std.testing.allocator;
    const filename = "converted_f32/model-00001-of-00001.safetensors";
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var reg: Registry = try .fromPath(a, io, filename);
    defer reg.deinit();

    var file = try File.open(io, a, filename);
    defer file.close();

    var r_it = reg.iterator();

    var buffer: [4096]u8 = undefined;

    while (r_it.next()) |r_val| {
        // Only look at f32 tensors
        if (r_val.value_ptr.dtype != .F32) continue;

        var r_tr: TensorReader = try .init(
            io,
            r_val.value_ptr,
            &buffer,
        );
        defer r_tr.deinit();

        var f_br = try file.tensor(
            f32,
            r_val.value_ptr.name,
        ) orelse return error.MissingTensor;
        var f_br_r = try f_br.reader(a);
        defer f_br_r.deinit();

        while (try f_br_r.next()) |value| {
            const other: u32 =
                try r_tr.interface.takeInt(u32, .little);
            try std.testing.expectEqual(other, @as(u32, @bitCast(value)));
        }
    }
}
