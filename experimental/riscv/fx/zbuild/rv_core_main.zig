// P0 stripped root: wasm_core_main.zig with one provider. The app's configured
// OpenAI Chat Completions provider (FX_CONFIGURED_PROVIDERS) is the only model
// route: no Codex, no Grok, no AI Gateway stream, no model catalogs or credits.
// Copied into fx/src next to wasm_core_main.zig at build time; fx's own files
// are unchanged.
const std = @import("std");
const build_options = @import("build_options");
const acp_server = @import("acp/server.zig");
const js_host_stream_provider = @import("gateway/js_host_stream_provider.zig");
const host_chat_completions = @import("gateway/host_chat_completions.zig");
const model_provider = @import("core/config/model_provider.zig");
const context_contract = @import("core/workspace/context_contract.zig");
const gateway_provider = @import("core/gateway/gateway_provider.zig");
const provider_set = @import("core/gateway/provider_set.zig");
const agent_steps = @import("core/config/agent_steps.zig");
const host = @import("core/hosts/host.zig");
const js_host_attachments = @import("core/hosts/js_host_attachments.zig");
const io_mod = @import("core/shared/io.zig");
const oauth_transport = @import("core/auth/oauth_transport.zig");
const builtin_gateway = @import("builtins/gateway.zig");
const builtin_modes = @import("builtins/modes.zig");

comptime {
    if (build_options.wasm_surface != .core) {
        @compileError("fx-core requires -Dwasm-surface=core");
    }
}

pub const panic = @import("core/hosts/wasm_panic.zig").panic;

pub fn main(init: std.process.Init) !void {
    io_mod.setIo(init.io);
    io_mod.setEnvironMap(init.environ_map);
    const alloc = std.heap.c_allocator;
    const json = io_mod.getenv("FX_CONFIGURED_PROVIDERS") orelse return error.InvalidConfiguredProvider;
    var custom_context = try host_chat_completions.Context.init(alloc, js_host_stream_provider.transport, json);
    defer custom_context.deinit(alloc);
    var providers = provider_set.gateway_only(.{});
    providers.configured_fn = host_chat_completions.bundle;
    providers.configured_context = &custom_context;
    try acp_server.run(alloc, .{
        .default_model = builtin_gateway.default_model,
        .default_agent_step_limit = agent_steps.default_max_agent_steps,
        .gateway_retry_count = 0,
        .gateway_chat_url = builtin_gateway.default_chat_url,
        .gateway_models_path = builtin_gateway.models_path,
        .gateway_provider = .{
            .oauth_transport = oauth_transport.unavailable_provider,
            .chat_url = .{ .resolve_fn = resolveChatUrl },
        },
        .provider_set = providers,
        .secret_store = host.unavailable_secret_store,
        .prompt_policy = .{ .system_prompt = "" },
        .ignored_list_entries = &.{},
        .max_list_entries = 0,
        .max_read_file_bytes = 0,
        .max_read_file_lines = 0,
        .max_read_file_line_len = 0,
        .max_command_output_bytes = 0,
        .max_tool_result_bytes = 64 * 1024,
        .max_history_turns = 100,
        .context_registry = .{ .default_provider = context_contract.empty_provider },
        .mode_registry = builtin_modes.registry,
        .credential_override = null,
        .provider_override = try providerOverride(io_mod.getenv("FX_PROVIDER"), &custom_context),
        .provider_auth_mode = .host_managed,
        .configured_providers_json = json,
        .model_override = io_mod.getenv("FX_MODEL"),
        .effort_override = null,
        .fast_override = null,
        .ultrafast_override = null,
        .workspace_root_override = "/",
        .allow_acp_mcp = false,
        .allow_native_tools = false,
        .minimal_kernel = true,
        .host_attachments = js_host_attachments.store,
    });
}

fn providerOverride(value: ?[]const u8, context: *const host_chat_completions.Context) error{InvalidProvider}!?model_provider.ProviderId {
    const raw = value orelse return null;
    const provider = model_provider.parse(raw) orelse return error.InvalidProvider;
    if (provider != .configured) return error.InvalidProvider;
    return provider.bind(context.registry) catch error.InvalidProvider;
}

fn resolveChatUrl(_: ?*anyopaque, fallback: []const u8) []const u8 {
    return fallback;
}
