"""REPL worker subprocess — runs inside a child process.

Protocol:
  stdin  <- line-delimited JSON: {"code": "...", "reset": false}
  fd=W   -> line-delimited JSON: {"stdout": "...", "error": null, "vars": [...]}

The write fd is passed via LIAGENT_REPL_FD environment variable.
"""
from __future__ import annotations

import builtins
import dis
import io
import json
import os
import sys
import traceback

_MAX_OUTPUT_BYTES = 10 * 1024  # 10KB
_RESULT_FD = int(os.environ.get("LIAGENT_REPL_FD", "3"))
_REPL_MODE = str(os.environ.get("LIAGENT_REPL_MODE", "sandboxed") or "sandboxed").strip().lower()

_BLOCKED_ATTRS = frozenset({
    "__class__", "__dict__", "__mro__", "__bases__", "__base__",
    "__subclasses__", "__getattribute__", "__getattr__", "__setattr__",
    "__delattr__", "__globals__", "__closure__", "__code__", "__defaults__",
    "__kwdefaults__", "__func__", "__self__", "__wrapped__", "__call__",
    "__reduce__", "__reduce_ex__", "__builtins__", "__import__", "__loader__",
    "__spec__", "__annotations__", "__dir__", "__traceback__", "tb_frame",
    "tb_next", "f_back", "f_globals", "f_locals", "f_builtins",
    "gi_frame", "cr_frame", "ag_frame", "_invoke", "_impl",
})

# ── Module guard: block at sys.modules level ─────────────────────────
# Only modules that Python's internal machinery does NOT depend on.
# os, pathlib, io are NOT here — they're blocked via __import__ in the
# exec namespace instead (see _build_sandbox_callables).
_SYSMOD_BLOCKED = frozenset({
    "socket", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc", "paramiko",
    "subprocess", "shlex", "pty", "pdb",
    "ctypes", "cffi", "signal", "resource",
    "importlib", "imp", "runpy", "code", "codeop", "compileall",
    "webbrowser", "antigravity",
    "shutil", "glob", "tempfile", "fileinput",
})

# Blocked via __import__ override in exec namespace.
# Python internals (traceback, linecache, etc.) keep their references.
_IMPORT_BLOCKED = frozenset({
    "os", "pathlib", "io", "sys", "inspect", "operator", "builtins",
})

_CWORK_ROOT = ""  # set in main() from env
# Captured before any guard — used by sandbox open
_realpath = os.path.realpath
_os_sep = os.sep
# Track sandbox builtins dict objects to identify user-origin frames,
# including nested eval/exec/compile frames with non-<repl> filenames.
_SANDBOX_BUILTINS: list[dict] = []
_GUARDED_BLOCKED: frozenset[str] = frozenset()
_INTERNAL_GUARD_FUNCS = frozenset({
    "_attr_guard", "_called_from_repl", "_is_sandbox_frame",
    "_guarded_open", "_guarded_import",
})


def _called_from_repl() -> bool:
    frame = sys._getframe(1)  # noqa: SLF001
    while frame is not None:
        if frame.f_code.co_filename == "<repl>":
            return True
        frame_builtins = frame.f_globals.get("__builtins__")
        if isinstance(frame_builtins, dict):
            for b in _SANDBOX_BUILTINS:
                if frame_builtins is b:
                    return True
        frame = frame.f_back
    return False


def _guarded_open(file, mode="r", *a, _raw_open=builtins.open, **kw):
    if _called_from_repl() and _CWORK_ROOT:
        p = file
        if isinstance(p, int):
            raise PermissionError(
                "File access denied: fd-based open not allowed in sandbox"
            )
        if hasattr(file, "__fspath__"):
            try:
                p = type(file).__fspath__(file)
            except Exception:
                raise PermissionError("File access denied: fspath failed")
        if isinstance(p, bytes):
            try:
                p = p.decode("utf-8")
            except Exception:
                raise PermissionError(
                    "File access denied: invalid path encoding"
                )
        if isinstance(p, str):
            resolved = _realpath(p)
            cw = _realpath(_CWORK_ROOT)
            if not resolved.startswith(cw + _os_sep) and resolved != cw:
                raise PermissionError(
                    "File access denied: "
                    + str(p) + " is outside workspace"
                )
            return _raw_open(p, mode, *a, **kw)
        raise PermissionError("File access denied: unsupported path type")
    return _raw_open(file, mode, *a, **kw)


def _guarded_import(name, *args, _raw_import=builtins.__import__, **kwargs):
    top = name.split('.')[0]
    if top in _GUARDED_BLOCKED and _called_from_repl():
        raise ImportError(f"Module '{name}' is not available")
    return _raw_import(name, *args, **kwargs)


def _build_sandbox_callables(orig_open, orig_import, all_blocked,
                             cwork_root, realpath_fn, sep,
                             raw_builtins_dict):
    """Build open(), __import__(), and sanitized builtins in an isolated namespace.

    Each wrapped callable is a unique class with __slots__ = () — NO instance
    storage at all. The wrapped target is dispatched through the class, so
    __call__.__closure__ does not expose target functions. This means
    object.__getattribute__(wrapper, '_f') raises AttributeError
    because there is no '_f' slot or attribute to find.

    C builtins are double-wrapped (Python function hides __self__, then
    class-wrapper hides instance attributes).
    """
    _exec_globals = {
        '__name__': '_sandbox',
        '__builtins__': {
            '__build_class__': __build_class__,
            'isinstance': isinstance,
            'callable': callable,
            'bytes': bytes,
            'str': str,
            'int': int,
            'hasattr': hasattr,
            'type': type,
            'object': object,
            'property': property,
            'PermissionError': PermissionError,
            'ImportError': ImportError,
            'AttributeError': AttributeError,
            'Exception': Exception,
        },
    }

    _exec_code = '''
def _wrap(f):
    """Create an opaque callable wrapper.

    Each call creates a UNIQUE class with empty __slots__. The wrapped
    function is stored as a class staticmethod and __call__ dispatches
    via type(self), so __call__.__closure__ does not expose wrapped funcs.
    There is no instance-level storage. This means:
      object.__getattribute__(wrapper, '_f') -> AttributeError
      wrapper._f                             -> AttributeError
    """
    class _WM(type):
        __slots__ = ()
        @property
        def _invoke(cls):
            raise AttributeError("attribute access denied")
        @property
        def _impl(cls):
            raise AttributeError("attribute access denied")

    class _W(metaclass=_WM):
        __slots__ = ()
        _impl = f
        @property
        def __class__(self):
            raise AttributeError("attribute access denied")
        def __call__(self, *a, **kw):
            cls_dict = type.__dict__['__dict__'].__get__(type(self), type)
            return cls_dict['_impl'](*a, **kw)
        def __getattribute__(self, name):
            raise AttributeError("attribute access denied")
        def __repr__(self):
            return "<built-in function>"
    return _W()

def _make_open(orig, cwork, rpath, sep):
    def _open(file, mode="r", *a, **kw):
        if cwork:
            p = file
            # Block fd-based open (integer file descriptors)
            if isinstance(p, int):
                raise PermissionError(
                    "File access denied: fd-based open not allowed in sandbox"
                )
            # Handle os.PathLike objects (prevents __fspath__ bypass)
            if hasattr(file, "__fspath__"):
                try:
                    p = type(file).__fspath__(file)
                except Exception:
                    raise PermissionError("File access denied: fspath failed")
            # Handle bytes paths
            if isinstance(p, bytes):
                try:
                    p = p.decode("utf-8")
                except Exception:
                    raise PermissionError(
                        "File access denied: invalid path encoding"
                    )
            # Validate string path against cwork boundary
            if isinstance(p, str):
                resolved = rpath(p)
                cw = rpath(cwork)
                if not resolved.startswith(cw + sep) and resolved != cw:
                    raise PermissionError(
                        "File access denied: "
                        + str(p) + " is outside workspace"
                    )
                # Open with validated string to prevent TOCTOU with PathLike
                return orig(p, mode, *a, **kw)
            # Unrecognized path type — deny
            raise PermissionError("File access denied: unsupported path type")
        return orig(file, mode, *a, **kw)
    return _wrap(_open)

def _make_import(orig, blocked):
    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top in blocked:
            raise ImportError(
                "Module \'" + name + "\' is not available in the REPL sandbox"
            )
        return orig(name, globals, locals, fromlist, level)
    return _wrap(_imp)

def _sanitize_builtins(raw, skip):
    """Wrap C builtins to hide __self__ (builtins module reference).

    C builtin functions (builtin_function_or_method) have __self__
    pointing to the builtins module. Wrapping in Python function + _wrap
    hides this: Python functions don't have __self__, and _wrap's
    closure-class has no instance storage.
    """
    safe = {}
    for name, obj in raw.items():
        if name in skip:
            continue
        if callable(obj) and hasattr(obj, '__self__'):
            def _make_w(fn):
                def _w(*a, **kw):
                    return fn(*a, **kw)
                return _wrap(_w)
            safe[name] = _make_w(obj)
        else:
            safe[name] = obj
    return safe
'''

    # Use single namespace (globals=locals) so _wrap is visible to all functions
    exec(compile(_exec_code, "<sandbox_wrappers>", "exec"), _exec_globals)  # noqa: S102

    safe_open = _exec_globals['_make_open'](orig_open, cwork_root, realpath_fn, sep)
    safe_import = _exec_globals['_make_import'](orig_import, all_blocked)
    sanitized = _exec_globals['_sanitize_builtins'](
        raw_builtins_dict, {'__import__', 'open'},
    )

    # Clean up: remove factory functions from globals to prevent
    # type(open).__call__.__globals__ from leaking references
    for key in ('_wrap', '_make_open', '_make_import', '_sanitize_builtins'):
        _exec_globals.pop(key, None)

    # Minimize exec namespace builtins to only what wrapper methods need
    # at runtime (__getattribute__ needs AttributeError). This closes the
    # secondary escape: type(open).__call__.__globals__['__builtins__']
    # no longer contains any C builtins with __self__.
    _exec_globals['__builtins__'] = {
        'AttributeError': AttributeError,
        'type': type,
    }

    return safe_open, safe_import, sanitized


def _install_sysmod_guard():
    """Block dangerous modules at sys.modules level."""
    for m in _SYSMOD_BLOCKED:
        sys.modules[m] = None


def _make_safe_builtins():
    """Create a restricted __builtins__ dict for exec namespace.

    This is the primary sandbox boundary for user code:
    - __import__ blocks os/pathlib/io AND all _SYSMOD_BLOCKED modules
    - open() restricts file access to cwork paths (handles str, bytes, PathLike)
    - All wrapped in closure-classes with no instance storage (empty __slots__)
    - C builtins double-wrapped to hide __self__ (builtins module reference)
    - object.__getattribute__(wrapper, anything) finds nothing to extract
    """
    all_blocked = _SYSMOD_BLOCKED | _IMPORT_BLOCKED

    safe_open, safe_import, sanitized = _build_sandbox_callables(
        # builtins.open / builtins.__import__ are guarded in main() before
        # namespace creation, so even if wrapper internals are recovered the
        # underlying primitives still enforce sandbox policy.
        orig_open=builtins.open,
        orig_import=builtins.__import__,
        all_blocked=all_blocked,
        cwork_root=_CWORK_ROOT,
        realpath_fn=_realpath,
        sep=_os_sep,
        raw_builtins_dict=dict(vars(builtins)),
    )

    sanitized["__import__"] = safe_import
    sanitized["open"] = safe_open
    _builtin_vars = vars

    # Guard dynamic attribute helpers (e.g., getattr(obj, "__dict__")).
    def _safe_getattr(obj, name, *default):
        if isinstance(name, str) and name in _BLOCKED_ATTRS:
            raise AttributeError("attribute access denied")
        if default:
            return getattr(obj, name, *default)
        return getattr(obj, name)

    def _safe_hasattr(obj, name):
        if isinstance(name, str) and name in _BLOCKED_ATTRS:
            return False
        return hasattr(obj, name)

    def _safe_vars(obj=None):
        # vars(type(open)) exposes wrapper internals in mappingproxy; filter them.
        if obj is None:
            return _builtin_vars()
        data = _builtin_vars(obj)
        try:
            items = dict(data).items()
        except Exception:
            return data
        return {k: v for k, v in items if k not in _BLOCKED_ATTRS}

    sanitized["getattr"] = _safe_getattr
    sanitized["hasattr"] = _safe_hasattr
    sanitized["vars"] = _safe_vars
    # Convenience: provide StringIO/BytesIO since io is blocked
    sanitized["StringIO"] = io.StringIO
    sanitized["BytesIO"] = io.BytesIO
    return sanitized


def _make_namespace():
    """Create a fresh exec namespace with sandboxed builtins."""
    safe_builtins = _make_safe_builtins()
    _SANDBOX_BUILTINS.append(safe_builtins)
    return {"__builtins__": safe_builtins}


def _send_result(pipe, result: dict):
    line = json.dumps(result, ensure_ascii=False, default=str) + "\n"
    pipe.write(line)
    pipe.flush()


def _run_code(namespace: dict, code: str) -> dict:
    old_stdout = sys.stdout
    capture = io.StringIO()
    sys.stdout = capture

    error = None
    old_trace = sys.gettrace()
    cur_frame = sys._getframe()  # noqa: SLF001
    old_f_trace = cur_frame.f_trace
    old_f_trace_opcodes = getattr(cur_frame, "f_trace_opcodes", False)

    ins_cache: dict[object, dict[int, dis.Instruction]] = {}

    def _ins_map(code_obj):
        m = ins_cache.get(code_obj)
        if m is None:
            m = {i.offset: i for i in dis.get_instructions(code_obj)}
            ins_cache[code_obj] = m
        return m

    def _is_sandbox_frame(frame) -> bool:
        # Internal wrapper helpers intentionally touch blocked attrs.
        if frame.f_code.co_filename == "<sandbox_wrappers>":
            return False
        if frame.f_code.co_name in _INTERNAL_GUARD_FUNCS:
            return False

        cur = frame
        while cur is not None:
            if cur.f_code.co_filename == "<repl>":
                return True
            frame_builtins = cur.f_globals.get("__builtins__")
            if isinstance(frame_builtins, dict):
                for b in _SANDBOX_BUILTINS:
                    if frame_builtins is b:
                        return True
            cur = cur.f_back
        return False

    def _attr_guard(frame, event, arg):
        if not _is_sandbox_frame(frame):
            return _attr_guard
        if event in {"call", "line"}:
            frame.f_trace_opcodes = True
            return _attr_guard
        if event == "opcode":
            ins = _ins_map(frame.f_code).get(frame.f_lasti)
            if ins and ins.opname in {"LOAD_ATTR", "LOAD_METHOD", "STORE_ATTR", "DELETE_ATTR"}:
                name = ins.argval
                if isinstance(name, str) and name in _BLOCKED_ATTRS:
                    raise AttributeError("attribute access denied")
        return _attr_guard

    use_attr_guard = _REPL_MODE != "trusted_local"
    try:
        if use_attr_guard:
            cur_frame.f_trace = _attr_guard
            cur_frame.f_trace_opcodes = True
            sys.settrace(_attr_guard)
        compiled = compile(code, "<repl>", "exec")
        exec(compiled, namespace)  # noqa: S102
    except Exception:
        error = traceback.format_exc()
    finally:
        if use_attr_guard:
            sys.settrace(old_trace)
            cur_frame.f_trace = old_f_trace
            cur_frame.f_trace_opcodes = old_f_trace_opcodes
        sys.stdout = old_stdout

    stdout = capture.getvalue()
    if len(stdout) > _MAX_OUTPUT_BYTES:
        stdout = stdout[:_MAX_OUTPUT_BYTES] + "\n...(truncated to 10KB)"

    user_vars = sorted(
        k for k in namespace
        if not k.startswith("_") and k not in {"__builtins__"}
    )

    return {
        "stdout": stdout,
        "error": error,
        "vars": user_vars[:50],
    }


def main():
    global _CWORK_ROOT, _GUARDED_BLOCKED
    _CWORK_ROOT = os.environ.get("LIAGENT_CWORK_ROOT", "")

    result_pipe = os.fdopen(_RESULT_FD, "w")
    if _REPL_MODE == "trusted_local":
        # Trusted mode: local REPL semantics, no sandbox guards.
        namespace = {"__builtins__": dict(vars(builtins))}
    else:
        _install_sysmod_guard()

        # Defense-in-depth: if user code recovers the builtins module object via
        # introspection, guard open/import only when the active call stack
        # originates from sandbox user frames. This avoids breaking
        # interpreter-internal imports and file reads.
        _GUARDED_BLOCKED = _SYSMOD_BLOCKED | _IMPORT_BLOCKED

        builtins.open = _guarded_open
        builtins.__import__ = _guarded_import
        namespace = _make_namespace()

    # Keep global guards active for worker lifetime.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _send_result(result_pipe, {"stdout": "", "error": f"Invalid JSON: {e}", "vars": []})
            continue

        code = cmd.get("code", "")
        reset = cmd.get("reset", False)

        if reset:
            if _REPL_MODE == "trusted_local":
                namespace = {"__builtins__": dict(vars(builtins))}
            else:
                namespace = _make_namespace()
            _send_result(result_pipe, {"stdout": "", "error": None, "vars": []})
            continue

        if not code.strip():
            _send_result(result_pipe, {"stdout": "", "error": None, "vars": list(namespace.keys())})
            continue

        result = _run_code(namespace, code)
        _send_result(result_pipe, result)


if __name__ == "__main__":
    main()
