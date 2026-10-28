const std = @import("std");

pub fn build(b: *std.Build) void {
    const root_module = b.addModule("nvfp4", .{
        .root_source_file = b.path("src/main.zig"),
        .target = b.graph.host,
    });
    const exe = b.addExecutable(.{
        .name = "nvfp4",
        .root_module = root_module,
    });

    b.installArtifact(exe);
}
