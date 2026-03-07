"""Tests for REPL worker subprocess protocol."""
import asyncio
import json
import os
import sys
import tempfile
import unittest


class ReplWorkerProtocolTests(unittest.TestCase):
    """Test the worker subprocess via actual spawn."""

    def _run(self, coro):
        return asyncio.run(coro)

    async def _spawn_and_run(self, code: str, reset: bool = False,
                              cwork_root: str = "") -> dict:
        """Spawn worker, send one command, read result, kill."""
        r_fd, w_fd = os.pipe()
        env = {
            **os.environ,
            "LIAGENT_REPL_FD": str(w_fd),
            "LIAGENT_CWORK_ROOT": cwork_root or tempfile.gettempdir(),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "liagent.tools.repl_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            pass_fds=(w_fd,),
        )
        os.close(w_fd)  # parent closes write end

        cmd = json.dumps({"code": code, "reset": reset}) + "\n"
        proc.stdin.write(cmd.encode())
        await proc.stdin.drain()

        reader = os.fdopen(r_fd, "r")
        loop = asyncio.get_event_loop()
        line = await asyncio.wait_for(
            loop.run_in_executor(None, reader.readline),
            timeout=10.0,
        )
        result = json.loads(line)

        proc.stdin.close()
        reader.close()
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return result

    def test_simple_expression(self):
        result = self._run(self._spawn_and_run("x = 42\nprint(x)"))
        self.assertIsNone(result.get("error"))
        self.assertIn("42", result.get("stdout", ""))
        self.assertIn("x", result.get("vars", []))

    def test_error_handling(self):
        result = self._run(self._spawn_and_run("1/0"))
        self.assertIsNotNone(result.get("error"))
        self.assertIn("ZeroDivision", result["error"])

    def test_blocked_import_subprocess(self):
        result = self._run(self._spawn_and_run("import subprocess"))
        self.assertIsNotNone(result.get("error"))

    def test_blocked_import_os(self):
        result = self._run(self._spawn_and_run("import os"))
        self.assertIsNotNone(result.get("error"))
        self.assertIn("import", result["error"].lower())

    def test_blocked_import_pathlib(self):
        result = self._run(self._spawn_and_run("import pathlib"))
        self.assertIsNotNone(result.get("error"))

    def test_blocked_import_shutil(self):
        result = self._run(self._spawn_and_run("import shutil"))
        self.assertIsNotNone(result.get("error"))

    def test_blocked_import_operator(self):
        result = self._run(self._spawn_and_run("import operator"))
        self.assertIsNotNone(result.get("error"))
        self.assertIn("import", result["error"].lower())

    def test_blocked_import_builtins(self):
        result = self._run(self._spawn_and_run("import builtins"))
        self.assertIsNotNone(result.get("error"))
        self.assertIn("import", result["error"].lower())

    def test_sys_modules_os_escape_blocked(self):
        """import sys + sys.modules['os'] must not expose os module."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_sysmodules_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "import sys\n"
            f"os_mod = sys.modules.get('os')\n"
            f"path = r'{marker}'\n"
            "if os_mod:\n"
            "    rc = os_mod.system('touch ' + path)\n"
            "    print('ESCAPED:' + str(rc))\n"
            "else:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNotNone(result.get("error"))
        self.assertIn("import", result["error"].lower())
        self.assertFalse(os.path.exists(marker))

    def test_open_inside_cwork_allowed(self):
        """open() should work for files inside cwork."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file inside the cwork dir
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello")
            code = f"f = open('{test_file}'); print(f.read()); f.close()"
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            self.assertIsNone(result.get("error"), result.get("error"))
            self.assertIn("hello", result.get("stdout", ""))

    def test_open_outside_cwork_blocked(self):
        """open() should deny access to files outside cwork."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = "f = open('/etc/passwd'); print(f.read()); f.close()"
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            self.assertIsNotNone(result.get("error"))
            self.assertIn("denied", result["error"].lower())

    def test_blocked_import_io_open_bypass(self):
        """io.open() must not bypass the sandbox (Bug #1 vector 1)."""
        result = self._run(self._spawn_and_run("import io; io.open('/etc/passwd')"))
        self.assertIsNotNone(result.get("error"))
        # io import itself should be blocked
        self.assertIn("import", result["error"].lower())

    def test_open_bytes_path_outside_cwork_blocked(self):
        """open(b'/etc/hosts') must be blocked by _restricted_open (Bug #1 vector 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = "f = open(b'/etc/hosts'); print(f.read()); f.close()"
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            self.assertIsNotNone(result.get("error"))
            self.assertIn("denied", result["error"].lower())

    def test_traceback_works_after_error(self):
        """traceback.format_exc() must work — os not broken in sys.modules (Bug #2)."""
        result = self._run(self._spawn_and_run("1/0"))
        self.assertIsNotNone(result.get("error"))
        # Must contain proper traceback, not a secondary import error
        self.assertIn("ZeroDivision", result["error"])
        self.assertNotIn("ModuleNotFoundError", result["error"])

    def test_open_globals_escape_blocked(self):
        """open.__globals__ must raise AttributeError, not leak module state."""
        code = "try:\n    g = open.__globals__\n    print('LEAKED')\nexcept AttributeError:\n    print('BLOCKED')"
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("LEAKED", result.get("stdout", ""))

    def test_open_globals_original_open_full_chain(self):
        """open.__globals__['_original_open'] full escape chain must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "try:\n"
                "    f = open.__globals__['_original_open']('/etc/hosts')\n"
                "    print('ESCAPED:' + f.read()[:10])\n"
                "except (AttributeError, KeyError, TypeError) as e:\n"
                "    print('BLOCKED:' + type(e).__name__)\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_import_globals_escape_blocked(self):
        """__import__.__globals__ must also be blocked."""
        code = (
            "imp = __builtins__['__import__']\n"
            "try:\n"
            "    g = imp.__globals__\n"
            "    print('LEAKED')\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))

    def test_traceback_frame_escape_blocked(self):
        """traceback/frame globals must not expose raw open/import primitives."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_traceback_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "try:\n"
            "    1/0\n"
            "except Exception as e:\n"
            "    try:\n"
            "        tb = e.__traceback__\n"
            "        frame = tb.tb_frame\n"
            "        print('LEAKED:' + str(bool(frame)))\n"
            "    except AttributeError:\n"
            "        print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertIn("BLOCKED", stdout)
        self.assertNotIn("LEAKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_open_pathlike_outside_cwork_blocked(self):
        """os.PathLike objects must be validated — __fspath__ bypass blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "class P:\n"
                "    def __fspath__(self):\n"
                "        return '/etc/hosts'\n"
                "try:\n"
                "    f = open(P())\n"
                "    print('ESCAPED:' + f.read()[:10])\n"
                "except PermissionError:\n"
                "    print('BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_open_pathlike_inside_cwork_allowed(self):
        """os.PathLike objects inside cwork should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("pathlike_ok")
            code = (
                f"class P:\n"
                f"    def __fspath__(self):\n"
                f"        return '{test_file}'\n"
                f"f = open(P())\n"
                f"print(f.read())\n"
                f"f.close()\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            self.assertIsNone(result.get("error"), result.get("error"))
            self.assertIn("pathlike_ok", result.get("stdout", ""))

    def test_open_slot_access_blocked(self):
        """open._f must be blocked — no slot exists (Round 5: empty __slots__)."""
        code = (
            "try:\n"
            "    f = open._f\n"
            "    print('LEAKED')\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("LEAKED", result.get("stdout", ""))

    def test_object_getattribute_slot_bypass_blocked(self):
        """object.__getattribute__(open, '_f') must fail — no _f slot exists (Round 5 Bug #1)."""
        code = (
            "try:\n"
            "    f = object.__getattribute__(open, '_f')\n"
            "    print('LEAKED:' + str(type(f)))\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("LEAKED", result.get("stdout", ""))

    def test_object_getattribute_len_bypass_blocked(self):
        """object.__getattribute__(len, '_f') must also fail for wrapped builtins."""
        code = (
            "try:\n"
            "    f = object.__getattribute__(len, '_f')\n"
            "    print('LEAKED:' + str(type(f)))\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("LEAKED", result.get("stdout", ""))

    def test_object_getattribute_open_full_escape_chain(self):
        """Full chain: object.__getattribute__ → any attr → read /etc/hosts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    for attr in ('_f', '_fn', 'func', '__wrapped__', '__func__'):\n"
                "        try:\n"
                "            val = object.__getattribute__(open, attr)\n"
                "            if callable(val):\n"
                "                f = val('/etc/hosts')\n"
                "                print('ESCAPED:' + f.read()[:10])\n"
                "                escaped = True\n"
                "                break\n"
                "        except AttributeError:\n"
                "            continue\n"
                "except Exception:\n"
                "    pass\n"
                "if not escaped:\n"
                "    print('BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_object_getattribute_len_to_import_chain(self):
        """Full chain: object.__getattribute__(len, '_f') → __self__ → __import__ (Round 5 Bug #2)."""
        code = (
            "escaped = False\n"
            "try:\n"
            "    inner = object.__getattribute__(len, '_f')\n"
            "    real_fn = inner.__closure__[0].cell_contents\n"
            "    mod = real_fn.__self__\n"
            "    mod.__import__('os')\n"
            "    escaped = True\n"
            "except (AttributeError, TypeError, ImportError):\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED", stdout)
        self.assertIn("BLOCKED", stdout)

    def test_open_slot_closure_chain_blocked(self):
        """open._f.__closure__ full escape chain must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "try:\n"
                "    inner = open._f\n"
                "    cells = inner.__closure__\n"
                "    for c in cells:\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            break\n"
                "except (AttributeError, TypeError) as e:\n"
                "    print('BLOCKED:' + type(e).__name__)\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_builtins_import_defense_in_depth(self):
        """Even if attacker recovers builtins module, __import__ should be restricted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "try:\n"
                "    obj_type = (42).__class__.__mro__[-1]\n"
                "    for sc in obj_type.__subclasses__():\n"
                "        try:\n"
                "            g = sc.__init__.__globals__\n"
                "            bi = g.get('__builtins__')\n"
                "            if bi is not None:\n"
                "                if hasattr(bi, '__import__'):\n"
                "                    m = bi.__import__('os')\n"
                "                    print('ESCAPED')\n"
                "                    break\n"
                "                elif isinstance(bi, dict) and '__import__' in bi:\n"
                "                    m = bi['__import__']('os')\n"
                "                    print('ESCAPED')\n"
                "                    break\n"
                "        except (AttributeError, TypeError, ImportError):\n"
                "            continue\n"
                "    else:\n"
                "        print('BLOCKED')\n"
                "except (AttributeError, TypeError, ImportError):\n"
                "    print('BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_len_self_builtins_leak_blocked(self):
        """len.__self__ must not expose builtins module (Round 4 Bug #2)."""
        code = (
            "try:\n"
            "    mod = len.__self__\n"
            "    print('LEAKED:' + str(type(mod)))\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("LEAKED", result.get("stdout", ""))

    def test_len_self_import_os_chain_blocked(self):
        """len.__self__.__import__('os') full escape chain must fail (Round 4 Bug #2 full chain)."""
        code = (
            "try:\n"
            "    mod = len.__self__\n"
            "    os_mod = mod.__import__('os')\n"
            "    print('ESCAPED')\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("BLOCKED", result.get("stdout", ""))
        self.assertNotIn("ESCAPED", result.get("stdout", ""))

    def test_print_self_builtins_leak_blocked(self):
        """print.__self__ must also be blocked (same vector as len)."""
        code = (
            "try:\n"
            "    mod = print.__self__\n"
            "    print('LEAKED')\n"
            "except AttributeError:\n"
            "    pass\n"
            "print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertIn("BLOCKED", stdout)
        self.assertNotIn("LEAKED", stdout)

    def test_type_open_call_globals_sanitized(self):
        """type(open).__call__.__globals__['__builtins__'] must not leak C builtins with __self__."""
        code = (
            "try:\n"
            "    g = type(open).__call__.__globals__\n"
            "    bi = g.get('__builtins__', {})\n"
            "    if isinstance(bi, dict):\n"
            "        for name, obj in bi.items():\n"
            "            if hasattr(obj, '__self__'):\n"
            "                print('LEAKED:' + name)\n"
            "                break\n"
            "        else:\n"
            "            print('CLEAN')\n"
            "    else:\n"
            "        print('LEAKED:module')\n"
            "except AttributeError:\n"
            "    print('CLEAN')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertIn("CLEAN", stdout)
        self.assertNotIn("LEAKED", stdout)

    def test_type_open_call_closure_escape_blocked(self):
        """type(open).__call__.__closure__ chain must not recover raw open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    call = type(open).__call__\n"
                "    cells = call.__closure__ or ()\n"
                "    if cells:\n"
                "        inner = cells[0].cell_contents\n"
                "        for c in (inner.__closure__ or ()):\n"
                "            obj = c.cell_contents\n"
                "            if callable(obj):\n"
                "                f = obj('/etc/hosts')\n"
                "                print('ESCAPED:' + f.read()[:10])\n"
                "                escaped = True\n"
                "                break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_eval_blocked_attr_access_still_blocked(self):
        """eval() generated frames must still be subject to blocked-attribute policy."""
        code = (
            "try:\n"
            "    eval('type.__dict__')\n"
            "    print('LEAKED')\n"
            "except AttributeError:\n"
            "    print('BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertIn("BLOCKED", stdout)
        self.assertNotIn("LEAKED", stdout)

    def test_open_class_invoke_escape_blocked(self):
        """object.__getattribute__(open, '__class__')._invoke chain must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    cls = object.__getattribute__(open, '__class__')\n"
                "    inner = cls._invoke\n"
                "    for c in (inner.__closure__ or ()):\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            escaped = True\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_import_class_invoke_escape_blocked(self):
        """object.__getattribute__(__import__, '__class__')._invoke must not recover raw import."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_import_invoke_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "escaped = False\n"
            "try:\n"
            "    imp = __builtins__['__import__']\n"
            "    cls = object.__getattribute__(imp, '__class__')\n"
            "    inner = cls._invoke\n"
            "    real_import = None\n"
            "    for c in (inner.__closure__ or ()):\n"
            "        obj = c.cell_contents\n"
            "        if callable(obj):\n"
            "            real_import = obj\n"
            "            break\n"
            "    if real_import is not None:\n"
            "        os_mod = real_import('os')\n"
            f"        rc = os_mod.system('touch {marker}')\n"
            "        print('ESCAPED:' + str(rc))\n"
            "        escaped = True\n"
            "except Exception:\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED:", stdout)
        self.assertIn("BLOCKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_type_open_invoke_escape_blocked(self):
        """type(open)._invoke chain must not recover raw open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    inner = type(open)._invoke\n"
                "    for c in (inner.__closure__ or ()):\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            escaped = True\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_type_import_invoke_escape_blocked(self):
        """type(__import__)._invoke must not recover raw __import__."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_type_import_invoke_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "escaped = False\n"
            "try:\n"
            "    imp = __builtins__['__import__']\n"
            "    inner = type(imp)._invoke\n"
            "    real_import = None\n"
            "    for c in (inner.__closure__ or ()):\n"
            "        obj = c.cell_contents\n"
            "        if callable(obj):\n"
            "            real_import = obj\n"
            "            break\n"
            "    if real_import is not None:\n"
            "        os_mod = real_import('os')\n"
            f"        rc = os_mod.system('touch {marker}')\n"
            "        print('ESCAPED:' + str(rc))\n"
            "        escaped = True\n"
            "except Exception:\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED:", stdout)
        self.assertIn("BLOCKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_type_dict_descriptor_impl_escape_blocked(self):
        """type.__dict__ descriptor path must not expose wrapper _impl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    d = type.__dict__['__dict__'].__get__(type(open), type)\n"
                "    inner = d['_impl']\n"
                "    for c in (inner.__closure__ or ()):\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            escaped = True\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_type_dict_descriptor_import_escape_blocked(self):
        """type.__dict__ descriptor path must not recover raw __import__."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_type_dict_import_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "escaped = False\n"
            "try:\n"
            "    imp = __builtins__['__import__']\n"
            "    d = type.__dict__['__dict__'].__get__(type(imp), type)\n"
            "    inner = d['_impl']\n"
            "    real_import = None\n"
            "    for c in (inner.__closure__ or ()):\n"
            "        obj = c.cell_contents\n"
            "        if callable(obj):\n"
            "            real_import = obj\n"
            "            break\n"
            "    if real_import is not None:\n"
            "        os_mod = real_import('os')\n"
            f"        rc = os_mod.system('touch {marker}')\n"
            "        print('ESCAPED:' + str(rc))\n"
            "        escaped = True\n"
            "except Exception:\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED:", stdout)
        self.assertIn("BLOCKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_vars_type_wrapper_internals_hidden(self):
        """vars(type(...)) must not expose wrapper internals like _impl/_invoke."""
        code = (
            "ok = True\n"
            "for fn in (open, __builtins__['__import__']):\n"
            "    d = vars(type(fn))\n"
            "    if '_impl' in d or '_invoke' in d:\n"
            "        ok = False\n"
            "print('CLEAN' if ok else 'LEAKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertIn("CLEAN", stdout)
        self.assertNotIn("LEAKED", stdout)

    def test_vars_inspect_getclosurevars_escape_blocked(self):
        """vars(type(...)) + inspect.getclosurevars chain must not escape."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_vars_inspect_marker")
        if os.path.exists(marker):
            os.remove(marker)
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    import inspect\n"
                "    targets = [open, __builtins__['__import__']]\n"
                "    for fn in targets:\n"
                "        d = vars(type(fn))\n"
                "        inner = d['_impl']\n"
                "        cv = inspect.getclosurevars(inner)\n"
                "        values = list(cv.nonlocals.values()) + list(cv.globals.values())\n"
                "        for obj in values:\n"
                "            if not callable(obj):\n"
                "                continue\n"
                "            try:\n"
                "                f = obj('/etc/hosts')\n"
                "                print('ESCAPED_OPEN:' + f.read()[:10])\n"
                "                escaped = True\n"
                "                break\n"
                "            except Exception:\n"
                "                pass\n"
                "            try:\n"
                "                m = obj('os')\n"
                f"                rc = m.system('touch {marker}')\n"
                "                print('ESCAPED_IMPORT:' + str(rc))\n"
                "                escaped = True\n"
                "                break\n"
                "            except Exception:\n"
                "                pass\n"
                "        if escaped:\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED_OPEN:", stdout)
            self.assertNotIn("ESCAPED_IMPORT:", stdout)
            self.assertIn("BLOCKED", stdout)
            self.assertFalse(os.path.exists(marker))

    def test_operator_attrgetter_open_escape_blocked(self):
        """operator.attrgetter('__dict__') chain must not recover raw open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    import operator\n"
                "    type_dict = operator.attrgetter('__dict__')(type)\n"
                "    desc = type_dict['__dict__']\n"
                "    d = desc.__get__(type(open), type)\n"
                "    inner = d['_impl']\n"
                "    for c in (operator.attrgetter('__closure__')(inner) or ()):\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            escaped = True\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_operator_attrgetter_import_escape_blocked(self):
        """operator.attrgetter('__closure__') chain must not recover raw __import__."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_operator_attrgetter_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "escaped = False\n"
            "try:\n"
            "    import operator\n"
            "    imp = __builtins__['__import__']\n"
            "    type_dict = operator.attrgetter('__dict__')(type)\n"
            "    desc = type_dict['__dict__']\n"
            "    d = desc.__get__(type(imp), type)\n"
            "    inner = d['_impl']\n"
            "    for c in (operator.attrgetter('__closure__')(inner) or ()):\n"
            "        obj = c.cell_contents\n"
            "        if callable(obj):\n"
            "            m = obj('os')\n"
            f"            rc = m.system('touch {marker}')\n"
            "            print('ESCAPED:' + str(rc))\n"
            "            escaped = True\n"
            "            break\n"
            "except Exception:\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED:", stdout)
        self.assertIn("BLOCKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_builtins_getattr_open_closure_escape_blocked(self):
        """import builtins + builtins.getattr(open, '__closure__') must not recover raw open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = (
                "escaped = False\n"
                "try:\n"
                "    import builtins\n"
                "    opn = builtins.getattr(builtins, 'open')\n"
                "    cells = builtins.getattr(opn, '__closure__') or ()\n"
                "    for c in cells:\n"
                "        obj = c.cell_contents\n"
                "        if callable(obj):\n"
                "            f = obj('/etc/hosts')\n"
                "            print('ESCAPED:' + f.read()[:10])\n"
                "            escaped = True\n"
                "            break\n"
                "except Exception:\n"
                "    pass\n"
                "print('ESCAPED' if escaped else 'BLOCKED')\n"
            )
            result = self._run(self._spawn_and_run(code, cwork_root=tmpdir))
            stdout = result.get("stdout", "")
            self.assertNotIn("ESCAPED:", stdout)
            self.assertIn("BLOCKED", stdout)

    def test_builtins_getattr_import_closure_escape_blocked(self):
        """import builtins + builtins.getattr(__import__, '__closure__') must not recover raw import."""
        marker = os.path.join(tempfile.gettempdir(), "repl_escape_builtins_getattr_marker")
        if os.path.exists(marker):
            os.remove(marker)
        code = (
            "escaped = False\n"
            "try:\n"
            "    import builtins\n"
            "    imp = builtins.getattr(builtins, '__import__')\n"
            "    cells = builtins.getattr(imp, '__closure__') or ()\n"
            "    for c in cells:\n"
            "        obj = c.cell_contents\n"
            "        if callable(obj):\n"
            "            m = obj('os')\n"
            f"            rc = m.system('touch {marker}')\n"
            "            print('ESCAPED:' + str(rc))\n"
            "            escaped = True\n"
            "            break\n"
            "except Exception:\n"
            "    pass\n"
            "print('ESCAPED' if escaped else 'BLOCKED')\n"
        )
        result = self._run(self._spawn_and_run(code))
        stdout = result.get("stdout", "")
        self.assertNotIn("ESCAPED:", stdout)
        self.assertIn("BLOCKED", stdout)
        self.assertFalse(os.path.exists(marker))

    def test_wrapped_builtins_still_work(self):
        """Wrapped C builtins (len, print, sorted, etc.) must still function correctly."""
        code = (
            "results = []\n"
            "results.append(len([1, 2, 3]) == 3)\n"
            "results.append(abs(-5) == 5)\n"
            "results.append(max(1, 2, 3) == 3)\n"
            "results.append(min(1, 2, 3) == 1)\n"
            "results.append(sorted([3, 1, 2]) == [1, 2, 3])\n"
            "results.append(list(range(3)) == [0, 1, 2])\n"
            "results.append(isinstance(42, int))\n"
            "results.append(sum([1, 2, 3]) == 6)\n"
            "if all(results):\n"
            "    print('ALL_OK')\n"
            "else:\n"
            "    print('FAIL:' + str(results))\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("ALL_OK", result.get("stdout", ""))

    def test_class_definition_still_works(self):
        """Class definitions must still work with wrapped __build_class__."""
        code = (
            "class Foo:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def double(self):\n"
            "        return self.x * 2\n"
            "f = Foo(21)\n"
            "print(f.double())\n"
        )
        result = self._run(self._spawn_and_run(code))
        self.assertIsNone(result.get("error"), result.get("error"))
        self.assertIn("42", result.get("stdout", ""))

    def test_output_truncation(self):
        result = self._run(self._spawn_and_run("print('x' * 20000)"))
        stdout = result.get("stdout", "")
        self.assertLessEqual(len(stdout), 10240 + 50)
