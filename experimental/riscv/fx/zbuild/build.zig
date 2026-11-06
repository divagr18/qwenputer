// Builds fx's headless core (or another root placed in fx/src) for the P0
// measurements without touching fx's own build.zig or sources.
//
//   zig build -Dfx=/abs/path/to/fx -Darch=wasm  -Doptimize=ReleaseSmall -Dstrip=false
//   zig build -Dfx=/abs/path/to/fx -Darch=rv32  -Doptimize=ReleaseSmall -Dcpu=generic_rv32+m+zba
//
// The module graph and flags mirror fx's addWasmArtifact (build.zig): same
// build_options, single-threaded, no stack checks/protector/unwind tables,
// session_manager as its own module. For rv32 the `extern "fx"` host imports
// resolve against host_stubs.c (a static libfx.a); on Qwenputer those would be
// ecalls into the host.
const std = @import("std");

const WasmSurface = enum { none, core, term };

pub fn build(b: *std.Build) void {
    const fx = b.option([]const u8, "fx", "absolute path of the fx checkout") orelse @panic("-Dfx required");
    const root = b.option([]const u8, "root", "root file inside fx/src") orelse "wasm_core_main.zig";
    const arch = b.option([]const u8, "arch", "wasm or rv32") orelse "wasm";
    const cpu = b.option([]const u8, "cpu", "rv32 cpu") orelse "generic_rv32+m+zba";
    const strip = b.option(bool, "strip", "strip symbols") orelse true;
    const name = b.option([]const u8, "name", "artifact name") orelse "fx-core";
    const optimize = b.standardOptimizeOption(.{});
    const is_wasm = std.mem.eql(u8, arch, "wasm");

    const target = if (is_wasm)
        b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .wasi })
    else
        b.resolveTargetQuery(std.Target.Query.parse(.{
            .arch_os_abi = "riscv32-linux-musl",
            .cpu_features = cpu,
        }) catch @panic("bad rv32 target"));

    const opts = b.addOptions();
    opts.addOption([]const u8, "git_commit", "757bdb9685bd");
    opts.addOption([]const u8, "app_version", "0.0.16");
    opts.addOption([]const u8, "update_channel", "stable");
    opts.addOption(WasmSurface, "wasm_surface", .core);
    opts.addOption(bool, "libfx", true);
    opts.addOption(bool, "builtin_tools", true);

    const src = b.pathJoin(&.{ fx, "src" });
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ src, root }) },
            .target = target,
            .optimize = optimize,
            .single_threaded = true,
            .link_libc = true,
            .stack_check = false,
            .stack_protector = false,
            .omit_frame_pointer = true,
            .unwind_tables = .none,
            .error_tracing = false,
            .strip = strip,
        }),
    });
    exe.stack_size = 1024 * 1024;
    exe.root_module.addImport("build_options", opts.createModule());

    const sm_opts = b.addOptions();
    sm_opts.addOption(bool, "hooks", false);
    sm_opts.addOption([]const u8, "trace_dir", "/nonexistent");
    sm_opts.addOption([]const u8, "src_dir", b.pathJoin(&.{ src, "core/session_manager" }));
    const sm = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ src, "core/session_manager/api.zig" }) },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .single_threaded = true,
    });
    sm.addImport("build_options", sm_opts.createModule());
    exe.root_module.addImport("session_manager", sm);

    if (!is_wasm) {
        const stubs = b.addLibrary(.{
            .name = "fx",
            .linkage = .static,
            .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true }),
        });
        stubs.root_module.addCSourceFile(.{ .file = b.path("host_stubs.c") });
        exe.root_module.linkLibrary(stubs);
        exe.link_gc_sections = true;
    }
    b.installArtifact(exe);
}
