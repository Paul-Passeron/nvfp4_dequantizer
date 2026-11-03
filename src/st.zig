const std = @import("std");

pub const DataType = enum {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    BF16,
    F8_E4M3,
    F8_E5M2,
    BOOL,

    pub fn str(self: DataType) []const u8 {
        return switch (self) {
            .U8 => "U8",
            .U16 => "U16",
            .U32 => "U32",
            .U64 => "U64",
            .I8 => "I8",
            .I16 => "I16",
            .I32 => "I32",
            .I64 => "I64",
            .F16 => "F16",
            .F32 => "F32",
            .F64 => "F64",
            .BF16 => "BF16",
            .F8_E4M3 => "F8_E4M3",
            .F8_E5M2 => "F8_E5M2",
            .BOOL => "BOOL",
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
        i8 => return .I8,
        i16 => return .I16,
        i32 => return .I32,
        i64 => return .I64,
        bool => return .BOOL,
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
        return @bitCast(std.mem.readInt(
            @Int(.unsigned, @bitSizeOf(T)),
            slice[0..@sizeOf(T)],
            .little,
        ));
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
    var r = try Registry.fromPath(allocator, io, filename);
    defer r.deinit();
    var t = r.get(tensor) orelse return error.DidntFindTensor;
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

    pub const Buffers = struct {
        file: []u8,
        interface: []u8 = &.{},
    };

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

    pub fn take(self: *TensorReader, n: usize) ![]const u8 {
        if (n > self.remaining) return error.EndOfStream;
        const bytes = try self.file_reader.interface.take(n);
        self.advance_by(n);
        return bytes;
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

    fn initInterface(buffer: []u8) std.Io.Reader {
        return .{
            .vtable = &.{
                .stream = stream,
                .discard = discard,
            },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
    }

    pub fn init(
        io: std.Io,
        tensor: *const Tensor,
        buffers: Buffers,
    ) !TensorReader {
        var file = try std.Io.Dir.cwd().openFile(io, tensor.path, .{
            .mode = .read_only,
        });
        errdefer file.close(io);

        var file_reader = file.reader(io, buffers.file);
        try file_reader.seekTo(tensor.offset);
        const interface = initInterface(buffers.interface);

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

    var reg = try Registry.fromPath(allocator, io, filename);
    defer reg.deinit();
    const t = reg.get(tensor) orelse return error.DidntFindTensor;

    const ts = std.testing;

    var buffer_file: [4096]u8 = undefined;
    var buffer_interface: [4096]u8 = undefined;
    var reader: TensorReader = try .init(io, t, .{
        .file = &buffer_file,
        .interface = &buffer_interface,
    });
    defer reader.deinit();
    var r = &reader.interface;
    const remaining = try r.allocRemaining(allocator, .unlimited);
    try ts.expectEqual(remaining.len, t.byte_len);
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
    ) Registry {
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
        var this: Registry = .init(allocator);
        if (!std.mem.endsWith(u8, path, ".safetensors")) {
            return error.UnsupportedFileKind;
        }
        try this.parseFile(io, path);
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
        const allocated_path = try self.arena.allocator().dupe(u8, path);
        while (it.next()) |entry| {
            // TODO: What to do with metadata ?
            // - Keep it as a raw JSON object ?
            // - Do something else ?
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

            const RawTensorInfo = struct {
                dtype: DataType,
                shape: []u64,
                data_offsets: [2]usize,
            };

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
                .path = allocated_path,
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
    var buffer_file: [4096]u8 = undefined;
    var buffer_interface: [4096]u8 = undefined;
    while (it.next()) |t| {
        var r: TensorReader =
            try .init(io, t.value_ptr, .{
                .file = &buffer_file,
                .interface = &buffer_interface,
            });
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
    var buffer_file: [4096]u8 = undefined;
    var buffer_interface: [4096]u8 = undefined;
    var w: std.Io.Writer.Discarding = .init(&.{});
    while (it.next()) |t| {
        var r: TensorReader =
            try .init(io, t.value_ptr, .{
                .file = &buffer_file,
                .interface = &buffer_interface,
            });
        defer r.deinit();
        const n = try r.interface.streamRemaining(&w.writer);
        try std.testing.expectEqual(t.value_ptr.byte_len, n);
    }
}
