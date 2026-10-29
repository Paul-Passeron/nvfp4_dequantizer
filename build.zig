const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const st = b.createModule(.{
        .root_source_file = b.path("src/st.zig"),
        .target = target,
        .optimize = optimize,
    });
    const nvfp4 = b.createModule(.{
        .root_source_file = b.path("src/nvfp4.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .module = st, .name = "st" },
        },
    });
    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .module = nvfp4, .name = "nvfp4" },
            .{ .module = st, .name = "st" },
        },
    });

    const exe = b.addExecutable(.{
        .name = "nvfp4",
        .root_module = root_module,
    });

    const exe_check = b.addExecutable(.{
        .name = "nvfp4",
        .root_module = root_module,
    });

    const check = b.step("check", "Check if it compiles");
    check.dependOn(&exe_check.step);

    b.installArtifact(exe);

    // Stolen from <https://github.com/goreleaser/example-zig/blob/main/build.zig>

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const nvfp4_unit_tests = b.addTest(.{
        .root_module = nvfp4,
    });

    const st_unit_tests = b.addTest(.{
        .root_module = st,
    });

    const run_nvfp4_unit_tests = b.addRunArtifact(nvfp4_unit_tests);
    const run_st_unit_tests = b.addRunArtifact(st_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = root_module,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_nvfp4_unit_tests.step);
    test_step.dependOn(&run_st_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
