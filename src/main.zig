const std = @import("std");

// Found the original NVIDIA paper at: <https://arxiv.org/pdf/2509.25149>

// From <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html>:
// The representation of an NVFP4 tensor element x is given by:
// `x = x_e2m1 * s_block * s_global`
// where
//
// `x_e2m1` is the 4-bit value,
// `s_block` is a local FP8 E4M3 scaling factor shared by a block of 16 consecutive elements,
// `s_global` is a global FP32 scaling factor applied to the entire tensor.

pub fn main() !void {
    std.debug.print("Hello, World !\n", .{});
}
