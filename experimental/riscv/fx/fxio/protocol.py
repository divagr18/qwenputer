"""The packet protocol between the RV32 guest and the outside world (docs/fx/io.md).

Two layers:

1. The call layer is what exists today. Every wasm import is an `ecall` (rv/gen_imports.py):
   a7 = 0x400 + import id, a0 = pointer to the u32 argument words (a u64 takes two, low first),
   a1 = word count, result in a0. Buffers cross by DMA into wasm linear memory.
   IMPORTS classifies each import: what has to happen to it for fx to run with no host.

2. The packet layer is what crosses between the chip and Qwen. Frames, little-endian:

       magic "FXIO" | version u8 | kind u8 | flags u16 | seq u32 | length u32 | payload

   REQUEST (guest to Qwen)  u32 url length, url, u32 headers length, headers JSON, body
   REPLY   (Qwen to guest)  u16 HTTP status, then each read as u32 length + bytes
   USER    (user to guest)  ACP client bytes for stdin
   TOOL_REQ / TOOL_RES      the host tools that have no in-guest code (grep_files, glob_files)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"FXIO"
VERSION = 1
HEADER = struct.Struct("<4sBBHII")
REQUEST, REPLY, USER, TOOL_REQ, TOOL_RES = 1, 2, 3, 4, 5
KIND_NAMES = {REQUEST: "REQUEST", REPLY: "REPLY", USER: "USER", TOOL_REQ: "TOOL_REQ", TOOL_RES: "TOOL_RES"}

OS, MODEL, STUB = "a", "b", "c"  # the brief's groups: OS-layer circuit, model I/O, dropped or stubbed

# name: (group, what it is on the way to no host)
IMPORTS = {
    # ---- (a) OS layer: constant or deterministic, no outside world ---------------------------
    "args_get": (OS, "ROM constant (argv = fx acp)"),
    "args_sizes_get": (OS, "ROM constant"),
    "environ_get": (OS, "ROM constant (provider selection, placeholder key)"),
    "environ_sizes_get": (OS, "ROM constant"),
    "clock_res_get": (OS, "ROM constant (1 ms)"),
    "clock_time_get": (OS, "frozen virtual clock: one constant (speed-iter row 5 measured no cost to fx)"),
    "random_get": (OS, "seeded PRNG in guest state; replay and circuit must agree"),
    "poll_oneoff": (OS, "park until the next packet; a clock subscription returns at once"),
    "proc_exit": (OS, "HALT"),
    "fd_read": (OS, "fd 0: USER packets through a ring buffer; fd >= 3: in-weights file store"),
    "fd_write": (OS, "fd 1 / 2: silent sink (ACP updates, stderr), never echoed to Qwen; fd >= 3: file store"),
    # ---- (a) OS layer: the file store (wasifs.py is the host-side stand-in) ------------------
    "fd_close": (OS, "file store: free an fd slot"),
    "fd_fdstat_get": (OS, "file store: fd metadata"),
    "fd_filestat_get": (OS, "file store: stat"),
    "fd_filestat_set_size": (OS, "file store: truncate"),
    "fd_pread": (OS, "file store: read at offset"),
    "fd_prestat_dir_name": (OS, "file store: ROM constant '/'"),
    "fd_prestat_get": (OS, "file store: ROM constant (one preopen)"),
    "fd_pwrite": (OS, "file store: write at offset"),
    "fd_readdir": (OS, "file store: directory walk"),
    "fd_seek": (OS, "file store: fd cursor"),
    "fd_sync": (OS, "file store: no-op, the store is the weights' state"),
    "path_create_directory": (OS, "file store: mkdir"),
    "path_filestat_get": (OS, "file store: stat by path"),
    "path_open": (OS, "file store: open (fx's edit path stages a file, then renames)"),
    "path_readlink": (OS, "file store: EINVAL (no symlinks in the store)"),
    "path_remove_directory": (OS, "file store: rmdir"),
    "path_rename": (OS, "file store: rename"),
    "path_unlink_file": (OS, "file store: unlink"),
    "fx_host_tool_call": (OS, "row 24: read / edit / write are fx's code in the guest (rv_tools_main.zig); "
                              "grep_files and glob_files still leave as TOOL_REQ, because fx's own do not "
                              "compile for WASM. Host work, disclosed, until the store has its own"),
    # ---- (b) model I/O --------------------------------------------------------------------------
    "fx_http_stream_open": (MODEL, "REQUEST packet: fx's whole request body, read out of guest memory"),
    "fx_http_stream_status": (MODEL, "REPLY's u16 status (always 200 today)"),
    "fx_http_stream_next": (MODEL, "REPLY's reads, DMA into fx's buffer"),
    "fx_http_stream_close": (MODEL, "end of the model call"),
    "fx_http_stream_retry_after": (MODEL, "constant -1: no retries from an in-model server"),
    # ---- (c) dropped or stubbed: fixed answers, fx has a code path for each ----------------------
    "fx_attachment_put": (STUB, "-1: no attachments"),
    "fx_attachment_size": (STUB, "-1: no attachments"),
    "fx_attachment_take": (STUB, "-1: no attachments"),
    "fx_host_tool_result_read": (STUB, "-1: results come back inline"),
    "fx_host_tool_result_release": (STUB, "no-op"),
    "fx_http_request": (STUB, "unavailable: fx-core only reaches it for non-model HTTP"),
    "fx_journal_flush": (STUB, "0: no journal is opened"),
    "fx_oauth_session_commit": (STUB, "unavailable: auth is a placeholder key"),
    "fx_oauth_session_load": (STUB, "unavailable"),
    "fx_oauth_session_remove": (STUB, "unavailable"),
    "fx_session_commit": (STUB, "unavailable: --no-save semantics, the session lives in the guest"),
    "fx_session_list": (STUB, "unavailable"),
    "fx_session_load": (STUB, "unavailable"),
    "fx_session_remove": (STUB, "unavailable"),
    "fx_steering_close": (STUB, "no-op"),
    "fx_steering_state": (STUB, "-1: no steering"),
    "fx_steering_take": (STUB, "0: no steering"),
    "fd_filestat_set_times": (STUB, "ENOSYS: mtimes are frozen"),
    "path_filestat_set_times": (STUB, "ENOSYS: mtimes are frozen"),
    "path_link": (STUB, "ENOSYS: no hard links in the store"),
    "path_symlink": (STUB, "ENOSYS: no symlinks in the store"),
}


def classify(imports: list[dict], calls: dict | None = None) -> list[dict]:
    """The table for an image's imports.json; every import must be classified."""
    rows = []
    for imp in imports:
        name = imp["name"]
        if name not in IMPORTS:
            raise KeyError(f"unclassified import {imp['module']}.{name}")
        group, what = IMPORTS[name]
        rows.append({"id": imp["id"], "module": imp["module"], "name": name, "group": group, "what": what,
                     "calls": (calls or {}).get(name, 0)})
    return rows


@dataclass
class Packet:
    kind: int
    seq: int
    payload: bytes
    flags: int = 0

    def encode(self) -> bytes:
        return HEADER.pack(MAGIC, VERSION, self.kind, self.flags, self.seq, len(self.payload)) + self.payload


def decode_stream(data: bytes) -> list[Packet]:
    out, i = [], 0
    while i < len(data):
        if len(data) - i < HEADER.size:
            raise ValueError("truncated header")
        magic, ver, kind, flags, seq, n = HEADER.unpack_from(data, i)
        if magic != MAGIC or ver != VERSION or kind not in KIND_NAMES:
            raise ValueError(f"bad frame at {i}")
        i += HEADER.size
        if len(data) - i < n:
            raise ValueError("truncated payload")
        out.append(Packet(kind, seq, data[i:i + n], flags))
        i += n
    return out


def _blob(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _take(p: bytes, i: int) -> tuple[bytes, int]:
    n = struct.unpack_from("<I", p, i)[0]
    return p[i + 4:i + 4 + n], i + 4 + n


def request(seq: int, url: bytes, headers: bytes, body: bytes) -> Packet:
    return Packet(REQUEST, seq, _blob(url) + _blob(headers) + body)


def parse_request(pkt: Packet) -> tuple[bytes, bytes, bytes]:
    url, i = _take(pkt.payload, 0)
    headers, i = _take(pkt.payload, i)
    return url, headers, pkt.payload[i:]


def reply(seq: int, chunks: list[bytes], status: int = 200) -> Packet:
    return Packet(REPLY, seq, struct.pack("<H", status) + b"".join(_blob(c) for c in chunks))


def parse_reply(pkt: Packet) -> tuple[int, list[bytes]]:
    status = struct.unpack_from("<H", pkt.payload, 0)[0]
    chunks, i = [], 2
    while i < len(pkt.payload):
        c, i = _take(pkt.payload, i)
        chunks.append(c)
    return status, chunks
