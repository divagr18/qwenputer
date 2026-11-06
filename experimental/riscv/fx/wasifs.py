"""A small WASI preview1 filesystem for host.py / rvhost.py: one preopen, fd 3 = "/",
mapped onto the scratch repository. It is the OS side of the native-tool runs
(speed-iter item 1): fx's own file tools run in the guest and reach the files only
through these calls, the way they would through any WASI runtime's preopen.

Only what fx's tools use is implemented; anything else returns ENOSYS and is counted in
`unsupported` so a run shows it.
"""
from __future__ import annotations

import os
import struct

ESUCCESS, EBADF, EEXIST, EINVAL, EISDIR, ENOENT, ENOSYS, ENOTDIR, ENOTEMPTY, EIO = 0, 8, 20, 28, 31, 44, 52, 54, 55, 29
FT_DIR, FT_REG = 3, 4
O_CREAT, O_DIRECTORY, O_EXCL, O_TRUNC = 1, 2, 4, 8
FD_APPEND = 1


class WasiFs:
    def __init__(self, host, root):
        self.h = host
        self.root = os.path.realpath(root)
        self.fds = {3: {"kind": "dir", "path": self.root}}
        self.next_fd = 4
        self.unsupported = {}
        self.calls = {}

    # ---- helpers -----------------------------------------------------------------------
    def _resolve(self, dirfd, p, n):
        d = self.fds.get(dirfd)
        if not d or d["kind"] != "dir":
            return None
        rel = self.h.text(p, n)
        full = os.path.normpath(os.path.join(d["path"], rel.lstrip("/") if dirfd == 3 else rel))
        if os.path.commonpath([full, self.root]) != self.root:
            return None
        return full

    def _filestat(self, path, out):
        try:
            st = os.stat(path)
        except OSError:
            return ENOENT
        ft = FT_DIR if os.path.isdir(path) else FT_REG
        b = struct.pack("<QQBxxxxxxxQQQQQ", 1, st.st_ino & 0xFFFFFFFFFFFFFFFF, ft, 1, st.st_size,
                        1_767_000_000_000_000_000, 1_767_000_000_000_000_000, 1_767_000_000_000_000_000)
        self.h.wr(out, b)
        return ESUCCESS

    def _iovs(self, iovs, n):
        return [struct.unpack("<II", self.h.rd(iovs + 8 * i, 8)) for i in range(n)]

    # ---- calls ---------------------------------------------------------------------------
    def prestat_get(self, fd, out):
        if fd != 3:
            return EBADF
        self.h.wr(out, struct.pack("<BxxxI", 0, 1))
        return ESUCCESS

    def prestat_dir_name(self, fd, p, n):
        if fd != 3:
            return EBADF
        self.h.wr(p, b"/"[:n])
        return ESUCCESS

    def fdstat_get(self, fd, out):
        f = self.fds.get(fd)
        if not f:
            return EBADF
        self.h.wr(out, struct.pack("<BxHxxxxQQ", FT_DIR if f["kind"] == "dir" else FT_REG, f.get("flags", 0),
                                   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF))
        return ESUCCESS

    def path_open(self, dirfd, dirflags, p, n, oflags, rights, rights_inh, fdflags, out):
        path = self._resolve(dirfd, p, n)
        if path is None:
            return ENOENT
        exists = os.path.exists(path)
        if oflags & O_DIRECTORY or (exists and os.path.isdir(path)):
            if not exists:
                return ENOENT
            if not os.path.isdir(path):
                return ENOTDIR
            fd = self.next_fd
            self.next_fd += 1
            self.fds[fd] = {"kind": "dir", "path": path}
            self.h.put32(out, fd)
            return ESUCCESS
        if exists and oflags & O_EXCL and oflags & O_CREAT:
            return EEXIST
        if not exists and not oflags & O_CREAT:
            return ENOENT
        if not os.path.isdir(os.path.dirname(path)):
            return ENOENT
        data = b"" if (not exists or oflags & O_TRUNC) else open(path, "rb").read()
        if not exists or oflags & O_TRUNC:
            open(path, "wb").close()
        fd = self.next_fd
        self.next_fd += 1
        self.fds[fd] = {"kind": "file", "path": path, "data": bytearray(data), "pos": 0, "flags": fdflags, "dirty": False}
        self.h.put32(out, fd)
        return ESUCCESS

    def close(self, fd):
        f = self.fds.pop(fd, None)
        if f is None:
            return EBADF
        self._flush(f)
        return ESUCCESS

    def _flush(self, f):
        if f["kind"] == "file" and f["dirty"]:
            with open(f["path"], "wb") as w:
                w.write(f["data"])
            f["dirty"] = False

    def read(self, fd, iovs, n, nread, offset=None):
        f = self.fds.get(fd)
        if not f:
            return EBADF
        if f["kind"] != "file":
            return EISDIR
        pos = f["pos"] if offset is None else offset
        total = 0
        for p, l in self._iovs(iovs, n):
            chunk = bytes(f["data"][pos:pos + l])
            self.h.wr(p, chunk)
            pos += len(chunk)
            total += len(chunk)
            if len(chunk) < l:
                break
        if offset is None:
            f["pos"] = pos
        self.h.put32(nread, total)
        return ESUCCESS

    def write(self, fd, iovs, n, nwritten, offset=None):
        f = self.fds.get(fd)
        if not f or f["kind"] != "file":
            return EBADF
        pos = len(f["data"]) if (offset is None and f.get("flags", 0) & FD_APPEND) else (f["pos"] if offset is None else offset)
        total = 0
        for p, l in self._iovs(iovs, n):
            b = self.h.rd(p, l)
            if pos > len(f["data"]):
                f["data"] += bytes(pos - len(f["data"]))
            f["data"][pos:pos + l] = b
            pos += l
            total += l
        if offset is None:
            f["pos"] = pos
        f["dirty"] = True
        self._flush(f)
        self.h.put32(nwritten, total)
        return ESUCCESS

    def seek(self, fd, off, whence, out):
        f = self.fds.get(fd)
        if not f or f["kind"] != "file":
            return EBADF
        if off >= 1 << 63:
            off -= 1 << 64
        base = {0: 0, 1: f["pos"], 2: len(f["data"])}.get(whence)
        if base is None or base + off < 0:
            return EINVAL
        f["pos"] = base + off
        self.h.wr(out, struct.pack("<Q", f["pos"]))
        return ESUCCESS

    def tell(self, fd, out):
        f = self.fds.get(fd)
        if not f or f["kind"] != "file":
            return EBADF
        self.h.wr(out, struct.pack("<Q", f["pos"]))
        return ESUCCESS

    def filestat_get(self, fd, out):
        f = self.fds.get(fd)
        if not f:
            return EBADF
        if f["kind"] == "file":
            self._flush(f)
        return self._filestat(f["path"], out)

    def path_filestat_get(self, dirfd, flags, p, n, out):
        path = self._resolve(dirfd, p, n)
        if path is None:
            return ENOENT
        return self._filestat(path, out)

    def filestat_set_size(self, fd, size):
        f = self.fds.get(fd)
        if not f or f["kind"] != "file":
            return EBADF
        del f["data"][size:]
        f["data"] += bytes(max(0, size - len(f["data"])))
        f["dirty"] = True
        self._flush(f)
        return ESUCCESS

    def readdir(self, fd, buf, buf_len, cookie, used):
        f = self.fds.get(fd)
        if not f or f["kind"] != "dir":
            return EBADF
        names = sorted(os.listdir(f["path"]))
        ents = [(".", FT_DIR), ("..", FT_DIR)] + [(nm, FT_DIR if os.path.isdir(os.path.join(f["path"], nm)) else FT_REG) for nm in names]
        out = bytearray()
        for i, (nm, ft) in enumerate(ents):
            if i < cookie:
                continue
            b = nm.encode()
            out += struct.pack("<QQIBxxx", i + 1, i + 1, len(b), ft) + b
            if len(out) >= buf_len:
                break
        out = out[:buf_len]
        self.h.wr(buf, bytes(out))
        self.h.put32(used, len(out))
        return ESUCCESS

    def mkdir(self, dirfd, p, n):
        path = self._resolve(dirfd, p, n)
        if path is None:
            return ENOENT
        if os.path.exists(path):
            return EEXIST
        os.mkdir(path)
        return ESUCCESS

    def unlink(self, dirfd, p, n):
        path = self._resolve(dirfd, p, n)
        if path is None or not os.path.exists(path):
            return ENOENT
        if os.path.isdir(path):
            return EISDIR
        os.unlink(path)
        return ESUCCESS

    def rmdir(self, dirfd, p, n):
        path = self._resolve(dirfd, p, n)
        if path is None or not os.path.exists(path):
            return ENOENT
        if os.listdir(path):
            return ENOTEMPTY
        os.rmdir(path)
        return ESUCCESS

    def readlink(self, dirfd, p, n, buf, buf_len, used):
        path = self._resolve(dirfd, p, n)
        if path is None or not os.path.lexists(path):
            return ENOENT
        if not os.path.islink(path):
            return EINVAL  # not a symbolic link, as POSIX readlink
        t = os.readlink(path).encode()[:buf_len]
        self.h.wr(buf, t)
        self.h.put32(used, len(t))
        return ESUCCESS

    def rename(self, fd1, p1, n1, fd2, p2, n2):
        a = self._resolve(fd1, p1, n1)
        b = self._resolve(fd2, p2, n2)
        if a is None or b is None or not os.path.exists(a):
            return ENOENT
        for f in self.fds.values():
            if f["kind"] == "file" and f["path"] == a:
                self._flush(f)
        os.replace(a, b)
        return ESUCCESS
