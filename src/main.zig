const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
const Io = std.Io;
const Clock = Io.Clock;
const print = std.debug.print;

test "nvfp4 and f32 are close enough" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var f = try st.File.open(io, a, "nvfp4/model.safetensors");
    defer f.close();

    var converted_f = try st.File.open(io, a, "converted_f32/model-00001-of-00001.safetensors");
    defer converted_f.close();

    var t = try nvfp4.NVFP4Tensor.from(
        &f,
        "model.layers.0.self_attn.q_proj",
    ) orelse return error.MissingLayer;
    var reader = try t.reader(a);
    defer reader.deinit();

    const ct = (try converted_f.tensor(
        f32,
        "model.layers.0.self_attn.q_proj.weight",
    )) orelse return error.MissingLayer;

    var c_reader = try ct.reader(a);
    defer c_reader.deinit();

    var out: [16]f32 = undefined;
    var converted_out: [16]f32 = undefined;

    while (true) {
        if (!try reader.next(&out)) break;
        for (&converted_out) |*cv| {
            const next = try c_reader.next();
            try std.testing.expect(next != null);
            cv.* = next.?;
        }
        for (out, converted_out) |actual, expected| {
            try std.testing.expectApproxEqRel(expected, actual, 1e-3);
        }
    }
}

fn print_names(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !void {
    var reg: st.Registry =
        try .fromPath(allocator, io, path);
    defer reg.deinit();
    var it = reg.iterator();
    while (it.next()) |t| {
        print("{s}\n", .{t.value_ptr.name});
    }
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const filename = "nvfp4/model.safetensors";

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    try print_names(
        allocator,
        io,
        filename,
    );
    print("\n\n", .{});
    try print_names(
        allocator,
        io,
        "converted_f32/model-00001-of-00001.safetensors",
    );
}
