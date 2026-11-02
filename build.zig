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
    const benchmarks = b.createModule(.{
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .module = nvfp4, .name = "nvfp4" },
            .{ .module = st, .name = "st" },
        },
    });
    const main = b.createModule(.{
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
        .root_module = main,
    });
    const bench = b.addExecutable(.{
        .name = "benchmarks",
        .root_module = benchmarks,
    });

    const exe_check = b.addExecutable(.{
        .name = "nvfp4",
        .root_module = main,
    });

    const bench_check = b.addExecutable(.{
        .name = "benchmarks",
        .root_module = benchmarks,
    });

    const check = b.step("check", "Check if it compiles");
    check.dependOn(&exe_check.step);
    check.dependOn(&bench_check.step);

    const exe_artifact = b.addInstallArtifact(exe, .{});
    const bench_artifact = b.addInstallArtifact(bench, .{});

    // Stolen from <https://github.com/goreleaser/example-zig/blob/main/build.zig>

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&exe_artifact.step);
    run_cmd.addPassthruArgs();

    const bench_cmd = b.addRunArtifact(bench);
    bench_cmd.step.dependOn(&bench_artifact.step);
    bench_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const bench_step = b.step("bench", "Run the benchmark");
    bench_step.dependOn(&bench_cmd.step);

    const nvfp4_unit_tests = b.addTest(.{
        .root_module = nvfp4,
    });
    const st_unit_tests = b.addTest(.{
        .root_module = st,
    });
    const bench_unit_tests = b.addTest(.{
        .root_module = benchmarks,
    });

    const run_nvfp4_unit_tests = b.addRunArtifact(nvfp4_unit_tests);
    const run_st_unit_tests = b.addRunArtifact(st_unit_tests);
    const run_bench_unit_tests = b.addRunArtifact(bench_unit_tests);
    const exe_unit_tests = b.addTest(.{
        .root_module = main,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_nvfp4_unit_tests.step);
    test_step.dependOn(&run_st_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
    test_step.dependOn(&run_bench_unit_tests.step);
}
