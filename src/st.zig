const std = @import("std");

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
