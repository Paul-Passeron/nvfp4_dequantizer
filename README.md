# NVFP4 Dequantizer

This is a multicore, SIMD NVFP4 dequantizer written in Zig.
It takes a `.safetensors` file as input and decodes its 4-bit NVFP4 weights to either f32, f16 or bf16.
Decoding is done on the fly via an `std.Io.Reader`.
A small cli tool is also available and outputs a dequantized `.safetensors` file.

## Requirements
Zig 0.17.0. Developed on macOS.

## Build

```sh
zig build -Doptimize=ReleaseFast
```

That installs `nvfp4` and `benchmarks` into `zig-out/bin/`.

## Usage

```sh
usage: nvfp4 [options] <input>

Options:
  -o, --output <path>           Output file                       (default: out.safetensors)
  -d, --datatype <f32|f16|bf16> Output type                       (default: f32)
      --e2m1 <tbl|lut|bits>     Set e2m1 dequantisation algorithm (default: tbl on aarch64, lut on other platforms)
      --e4m3 <lut|bits>         Set e4m3 dequantisation algorithm (default: lut)
      --dequant <simd|seq>      Set dequantization parallelism    (default: simd)
  -h, --help                    Show this help
```

Example:
```sh
zig-out/bin/nvfp4 nvfp4/model.safetensors -o model_bf16.safetensors -d bf16 --e2m1=lut
```

The nvfp4 groups of `<name>.weight_packed`, `<name>.weight_scale` and `<name>.weight_global_scale` tensors are converted into a single `<name>.weight` tensor.

All other tensors are copied, their datatype not being changed.

Input and outputs must be different files.

## Library

```zig
var registry = try st.Registry.fromPath(allocator, io, "model.safetensors");
defer registry.deinit();

const tensor = try nvfp4.Tensor.fromRegistry(
    .F16,
    &registry,
    "model.layers.0.self_attn.q_proj",
) orelse return error.NotNvfp4;

var packed_buffer: [32 * 1024]u8 = undefined;
var scale_buffer: [32 * 1024]u8 = undefined;

var reader = try tensor.reader(allocator, io, .{
    .packed_bytes = &packed_buffer,
    .scales = &scale_buffer,
}, .{ .dequant = .simd });
defer reader.deinit();

_ = try reader.interface.streamRemaining(writer);
```

The models aren't included in the repo.
Tests and benchmarks use `nvfp4/model.safetensors`, and the tests also compare against `converted_f32/model-00001-of-00001.safetensors` which was computed from `nvfp4/model.safetensors`.
