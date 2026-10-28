const std = @import("std");

pub fn build(b: *std.Build) void {
    const nvfp4 = b.addModule("nvfp4", .{
        .root_source_file = b.path("src/nvfp4.zig"),
        .target = b.graph.host,
    });
    const st = b.addModule("st", .{
        .root_source_file = b.path("src/st.zig"),
        .target = b.graph.host,
    });
    const root_module = b.addModule("main", .{
        .root_source_file = b.path("src/main.zig"),
        .target = b.graph.host,
        .imports = &.{
            .{ .module = nvfp4, .name = "nvfp4" },
            .{ .module = st, .name = "st" },
        },
    });
    const exe = b.addExecutable(.{
        .name = "nvfp4",
        .root_module = root_module,
    });

    b.installArtifact(exe);
}
