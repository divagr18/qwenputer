// Native-tool root (speed-iter item 1): fx's own wasm root, unchanged, plus one export
// that runs fx's built-in file tools (src/tools/filesystem and the core file-mutation
// path) inside the guest over WASI file access.
//
// Why a bridge and not `.allow_native_tools = true`: on WASI fx compiles its native tool
// set out of the ACP server (acp/tool_call_presentation.zig `activeToolSet` returns
// `tool_set_contract.empty` under `comptime host_target.is_wasm`, and acp/prompt.zig
// `hostToolProvider` always returns `js_host_tools.provider()` there). Any non-WASI OS
// tag takes fx's threaded native paths, which riscv32 cannot compile (u64 atomics,
// Thread.spawn). So the ACP server keeps libfx's host-tool contract, and the platform's
// implementation of `fx_host_tool_call` calls back into this export: the tool work runs
// as fx's exact native tool code, on the RV32 CPU, not in the host.
//
// What does not run this way: the agent runtime's native-tool admission around a call
// (permission review, read tracker, change tracker, file snapshots for undo). The host
// tool path runs instead, as it does in every libfx embedding.
//
// Copied into fx/src next to wasm_core_main.zig at build time; fx's files are unchanged.
const std = @import("std");
const core = @import("wasm_core_main.zig");
const builtin_tools = @import("builtins/tools.zig");
const tool_dispatch = @import("core/tooling/tool_dispatch.zig");
const tool_admission = @import("core/tooling/tool_admission.zig");
const file_mutation_execution = @import("core/tooling/file_mutation_execution.zig");
const types = @import("core/shared/types.zig");

pub const panic = core.panic;

pub fn main(init: std.process.Init) !void {
    return core.main(init);
}

/// The repository, under the WASI preopen "/". (Not fx-core's own "/": with root "/" fx's
/// display paths lose their first character, "edited inycalc/calc.py".)
const workspace_root = "/w";

/// fx's file tools as registered in builtins/tools.zig. builtin_tools.registry would pull
/// in the shell and web tools, which do not compile for WASI. grep_files and glob_files
/// do not compile either: both list files through core/workspace/tool_files.zig ->
/// core/indexer/scan.zig, which may start std.Thread workers, and fx's WASM builds are
/// single-threaded ("Cannot spawn thread when building in single-threaded mode"). They
/// stay host tools; fxplat_native_tool returns `not_native` for them.
const file_tools = [_]tool_dispatch.Tool{
    builtin_tools.read_file,
    builtin_tools.edit_file,
    builtin_tools.write_file,
};
const registry = tool_dispatch.Registry{ .tools = file_tools[0..] };

const Outcome = struct { ok: bool, body: []const u8 };

/// Same signature and result convention as libfx's `fx_host_tool_call` import
/// (core/hosts/js_host_tools.zig): returns the result length, or -3 when it does not
/// fit; status 0 = success, 1 = failure. `not_native` (-100): not one of the tools above,
/// so the platform hands the call to the host as before.
const not_native: i32 = -100;
export fn fxplat_native_tool(
    name_ptr: [*]const u8,
    name_len: usize,
    args_ptr: [*]const u8,
    args_len: usize,
    out_ptr: [*]u8,
    out_cap: usize,
    status_ptr: *u8,
    id_ptr: [*]const u8,
    id_len: usize,
) i32 {
    if (registry.lookup(name_ptr[0..name_len]) == null) return not_native;
    const alloc = std.heap.c_allocator;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const call = types.ToolCall{
        .id = id_ptr[0..id_len],
        .name = name_ptr[0..name_len],
        .arguments_json = args_ptr[0..args_len],
    };
    const outcome = run(arena, call) catch |err| Outcome{ .ok = false, .body = @errorName(err) };
    if (outcome.body.len > out_cap) return -3;
    @memcpy(out_ptr[0..outcome.body.len], outcome.body);
    status_ptr.* = if (outcome.ok) 0 else 1;
    return @intCast(outcome.body.len);
}

fn run(arena: std.mem.Allocator, call: types.ToolCall) !Outcome {
    if (std.mem.eql(u8, call.name, "edit_file") or std.mem.eql(u8, call.name, "write_file")) {
        const preflight = try tool_admission.preflightFileMutation(arena, call, .{
            .tool_registry = registry,
            .workspace_root = workspace_root,
            .permission_rules = .{},
            .permission_mode = .yolo,
            .can_prompt = false,
        });
        const authorization = switch (preflight) {
            .allowed, .prompt, .permission_required => |value| value,
            .policy_denied => return .{ .ok = false, .body = "file mutation denied by policy" },
            .tool_failure => |reason| return .{ .ok = false, .body = reason },
        };
        // execute() wants distinct call and result owners.
        var result_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        const result = try file_mutation_execution.execute(.{
            .call_allocator = arena,
            .result_allocator = result_arena.allocator(),
            .call = call,
            .authorization = authorization,
        });
        const body = try arena.dupe(u8, result.model_output);
        result_arena.deinit();
        return .{ .ok = result.status == .success, .body = body };
    }
    const ctx = tool_dispatch.DispatchContext{
        .allocator = arena,
        .workspace_root = workspace_root,
        .permission_mode = .yolo,
    };
    const result = try tool_dispatch.dispatchAuthorizedToolCall(ctx, registry, call);
    return .{ .ok = result.status == .success, .body = result.body };
}
