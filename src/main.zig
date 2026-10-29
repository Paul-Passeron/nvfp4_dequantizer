const std = @import("std");
const nvfp4 = @import("nvfp4");
const st = @import("st");
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

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var f = try st.File.open(
        io,
        allocator,
        "nvfp4/model.safetensors",
    );
    defer f.close();

    var converted_f = try st.File.open(
        io,
        allocator,
        "converted_f32/model-00001-of-00001.safetensors",
    );
    defer converted_f.close();

    var it = converted_f.iterator();
    while (it.next()) |val| {
        print("{s} {}\n", .{ val.name, val.dtype });
    }

    var t = try nvfp4.NVFP4Tensor.from(
        &f,
        "model.layers.0.self_attn.q_proj",
    ) orelse return error.MissingLayer;
    var reader = try t.reader(allocator);
    defer reader.deinit();

    const ct = (try converted_f.tensor(
        f32,
        "model.layers.0.self_attn.q_proj.weight",
    )) orelse return error.MissingLayer;

    var c_reader = try ct.reader(allocator);
    defer c_reader.deinit();

    var out: [16]f32 = undefined;
    var converted_out: [16]f32 = undefined;

    while (try reader.next(&out)) {
        for (&converted_out) |*cv| {
            if (try c_reader.next()) |x| {
                cv.* = x;
            } else {
                break;
            }
        }
        for (out, converted_out) |v, cv| {
            print("({} {}) ", .{ v, cv });
        }
    }
    print("\n", .{});
}
