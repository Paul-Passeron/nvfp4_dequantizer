const std = @import("std");
const st = @import("st");
const nvfp4 = @import("nvfp4");

pub const Destination = union(enum) {
    /// length must be `tensor.byteLen()`
    mem: []u8,
    file: struct {
        buffer: []u8,
        file: *const std.Io.File,
        offset: u64,
    },
    discard,
};

pub const Entry = union(enum) {
    nvfp4: nvfp4.Tensor,
    regular: *const st.Tensor,

    pub fn byteLen(self: *const Entry) usize {
        return switch (self.*) {
            .nvfp4 => |t| t.dequantByteLen(),
            .regular => |t| t.byte_len,
        };
    }

    pub fn name(self: *const Entry) []const u8 {
        return switch (self.*) {
            .nvfp4 => |t| std.mem.cutSuffix(u8, t.weight_packed.name, ".weight_packed").?,
            .regular => |t| t.name,
        };
    }

    pub fn dtype(self: *const Entry) st.DataType {
        return switch (self.*) {
            .nvfp4 => .F32,
            .regular => |t| t.dtype,
        };
    }
};

pub const Job = struct {
    tensor: Entry,

    dest: Destination,

    err: ?anyerror = null,

    fn run(self: *Job, io: std.Io, config: nvfp4.DequantConfig) void {
        self.runImpl(io, config) catch |err| {
            self.err = err;
            return;
        };
    }

    pub fn runImpl(self: *Job, io: std.Io, config: nvfp4.DequantConfig) !void {
        switch (self.tensor) {
            .regular => |t| try self.runRegular(t, io),
            .nvfp4 => |t| try self.runNVFP4(t, io, config),
        }
    }

    inline fn readIntoDest(io: std.Io, reader: *std.Io.Reader, dest: Destination) !void {
        switch (dest) {
            .mem => |buffer| {
                var w = std.Io.Writer.fixed(buffer);
                _ = try reader.streamRemaining(&w);
                try w.flush();
            },
            .discard => {
                var buffer: [32 * 1024]u8 = undefined;
                var w = std.Io.Writer.Discarding.init(&buffer);
                _ = try reader.streamRemaining(&w.writer);
            },
            .file => |f| {
                var w = f.file.writer(io, f.buffer);
                w.pos = f.offset;
                _ = try reader.streamRemaining(&w.interface);
                try w.flush();
            },
        }
    }

    inline fn runRegular(
        self: *Job,
        t: *const st.Tensor,
        io: std.Io,
    ) !void {
        var file_buffer: [32 * 1024]u8 = undefined;
        var interface_buffer: [32 * 1024]u8 = undefined;
        var reader = try st.TensorReader.init(io, t, .{
            .file = &file_buffer,
            .interface = &interface_buffer,
        });
        defer reader.deinit();
        try readIntoDest(io, &reader.interface, self.dest);
    }

    pub fn runNVFP4(self: *Job, t: nvfp4.Tensor, io: std.Io, config: nvfp4.DequantConfig) !void {
        var packed_buffer: [32 * 1024]u8 = undefined;
        var scale_buffer: [32 * 1024]u8 = undefined;

        var reader = try t.reader(io, .{
            .packed_bytes = &packed_buffer,
            .scales = &scale_buffer,
        }, config);
        defer reader.deinit();
        try readIntoDest(io, &reader.interface, self.dest);
    }
};

pub fn runAllJobs(io: std.Io, config: nvfp4.DequantConfig, jobs: []Job) !void {
    var group: std.Io.Group = .init;
    for (jobs) |*job|
        group.async(io, Job.run, .{ job, io, config });
    try group.await(io);
    for (jobs) |job| if (job.err) |err| return err;
}
