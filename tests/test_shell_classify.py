"""Tests for shell command classification and argument validation."""
import unittest


class ClassifyCommandTests(unittest.TestCase):
    """Test 3-tier command classification."""

    def test_safe_ls(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["ls"])
        self.assertEqual(tier, "safe")

    def test_safe_git_status(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["git", "status"])
        self.assertEqual(tier, "safe")

    def test_dev_git_commit(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["git", "commit", "-m", "test"])
        self.assertEqual(tier, "dev")

    def test_dev_pip_install(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["pip", "install", "requests"])
        self.assertEqual(tier, "dev")

    def test_privileged_rm(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["rm", "file.txt"])
        self.assertEqual(tier, "privileged")

    def test_privileged_sudo(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["sudo", "ls"])
        self.assertEqual(tier, "privileged")

    def test_unknown_command_denied(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["curl", "http://example.com"])
        self.assertEqual(tier, "denied")
        self.assertIn("not in allowlist", reason)

    def test_empty_argv_denied(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command([])
        self.assertEqual(tier, "denied")

    def test_env_denied(self):
        """env leaks API keys — must not be in safe tier."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["env"])
        self.assertEqual(tier, "denied")

    def test_find_exec_blocked(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["find", ".", "-exec", "rm", "{}", ";"])
        self.assertEqual(tier, "denied")
        self.assertIn("-exec", reason)

    def test_find_without_exec_safe(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["find", ".", "-name", "*.py"])
        self.assertEqual(tier, "safe")

    def test_safe_rg(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["rg", "AgentBrain", "src"])
        self.assertEqual(tier, "safe")

    def test_rg_preprocessor_denied(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["rg", "--pre", "cat", "needle", "src"])
        self.assertEqual(tier, "denied")
        self.assertIn("--pre", reason)

    def test_safe_sed_read_only(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["sed", "-n", "1,5p", "README.md"])
        self.assertEqual(tier, "safe")

    def test_sed_in_place_denied(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["sed", "-i", "", "s/a/b/", "README.md"])
        self.assertEqual(tier, "denied")

    def test_safe_git_show(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["git", "show", "HEAD~1"])
        self.assertEqual(tier, "safe")

    def test_dev_uv_sync(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["uv", "sync"])
        self.assertEqual(tier, "dev")

    def test_dev_pnpm_install(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["pnpm", "install"])
        self.assertEqual(tier, "dev")

    def test_dev_ruff_check(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["ruff", "check", "src"])
        self.assertEqual(tier, "dev")

    def test_dev_python_module(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["python", "-m", "pytest", "tests"])
        self.assertEqual(tier, "dev")

    def test_dev_npx_package(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["npx", "-y", "playwright", "test"])
        self.assertEqual(tier, "dev")

    def test_python_c_denied(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["python", "-c", "print(1)"])
        self.assertEqual(tier, "denied")
        self.assertIn("-c", reason)


class SafeCworkTests(unittest.TestCase):
    """Test that cwork-sandboxed filesystem ops are safe tier."""

    def test_mkdir_inside_cwork_safe(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["mkdir", "project"])
        self.assertEqual(tier, "safe")

    def test_cp_inside_cwork_safe(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["cp", "a.txt", "b.txt"])
        self.assertEqual(tier, "safe")

    def test_mv_inside_cwork_safe(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["mv", "old.txt", "new.txt"])
        self.assertEqual(tier, "safe")

    def test_touch_inside_cwork_safe(self):
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["touch", "file.txt"])
        self.assertEqual(tier, "safe")

    def test_mkdir_outside_cwork_denied(self):
        """mkdir targeting outside cwork must be denied, not dev (Bug #4)."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["mkdir", "/tmp/escape"])
        self.assertEqual(tier, "denied")

    def test_cp_outside_cwork_denied(self):
        """cp with source outside cwork must be denied (Bug #4)."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["cp", "/etc/passwd", "here.txt"])
        self.assertEqual(tier, "denied")

    def test_mkdir_no_args_safe(self):
        """mkdir with no args (will fail at runtime, but classification is safe)."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["mkdir"])
        self.assertEqual(tier, "safe")


class ValidatePathArgTests(unittest.TestCase):
    """Test path boundary enforcement."""

    def test_relative_path_ok(self):
        from liagent.tools.shell_classify import validate_path_arg
        ok, reason = validate_path_arg("subdir/file.txt")
        self.assertTrue(ok)

    def test_absolute_outside_cwork_denied(self):
        from liagent.tools.shell_classify import validate_path_arg
        ok, reason = validate_path_arg("/etc/passwd")
        self.assertFalse(ok)
        self.assertIn("cwork", reason.lower())

    def test_dotdot_escape_denied(self):
        from liagent.tools.shell_classify import validate_path_arg
        ok, reason = validate_path_arg("../../etc/passwd")
        self.assertFalse(ok)

    def test_home_tilde_denied(self):
        from liagent.tools.shell_classify import validate_path_arg
        ok, reason = validate_path_arg("~/.ssh/id_rsa")
        self.assertFalse(ok)


class ValidateArgvTests(unittest.TestCase):
    """Test command-aware argument validation."""

    def test_grep_pattern_not_treated_as_path(self):
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["ERROR", "app.log"])
        self.assertTrue(ok)

    def test_cat_outside_cwork_denied(self):
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("cat", ["/etc/passwd"])
        self.assertFalse(ok)

    def test_ls_no_args_ok(self):
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("ls", [])
        self.assertTrue(ok)

    def test_grep_flag_f_path_escape_denied(self):
        """grep -f /etc/passwd a.txt — flag-value path must be caught (Bug #3)."""
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["-f", "/etc/passwd", "a.txt"])
        self.assertFalse(ok)
        self.assertIn("path", reason.lower())

    def test_grep_flag_f_inside_cwork_ok(self):
        """grep -f patterns.txt a.txt — flag-value inside cwork is fine."""
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["-f", "patterns.txt", "a.txt"])
        self.assertTrue(ok)

    def test_flag_value_tilde_escape_denied(self):
        """Any flag-value with ~ path escape should be denied."""
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["-f", "~/.ssh/id_rsa", "a.txt"])
        self.assertFalse(ok)

    def test_flag_value_dotdot_escape_denied(self):
        """Any flag-value with ../ path escape should be denied."""
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["-f", "../../etc/passwd", "a.txt"])
        self.assertFalse(ok)

    def test_equals_flag_value_escape_denied(self):
        """--file=/etc/passwd must be caught (Bug #3 vector: equals-separated)."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["grep", "--file=/etc/passwd", "a.txt"])
        self.assertEqual(tier, "denied")
        self.assertIn("path", reason.lower())

    def test_relative_dot_traversal_escape_denied(self):
        """./../../etc/passwd must be caught (Bug #3 vector: dot-slash traversal)."""
        from liagent.tools.shell_classify import classify_command
        tier, reason = classify_command(["grep", "--file", "./../../etc/passwd", "a.txt"])
        self.assertEqual(tier, "denied")

    def test_equals_flag_inside_cwork_ok(self):
        """--file=./local.txt should be allowed (inside cwork)."""
        from liagent.tools.shell_classify import validate_argv
        ok, reason = validate_argv("grep", ["--file=./local.txt", "a.txt"])
        self.assertTrue(ok)


class GrantKeyTests(unittest.TestCase):
    """Test grant key generation."""

    def test_dev_git_grant_key(self):
        from liagent.tools.shell_classify import grant_key
        self.assertEqual(grant_key(["git", "commit"]), "shell_exec:dev:git")

    def test_dev_pip_grant_key(self):
        from liagent.tools.shell_classify import grant_key
        self.assertEqual(grant_key(["pip", "install", "flask"]), "shell_exec:dev:pip")

    def test_scoped_dev_grant_key(self):
        from liagent.tools.shell_classify import grant_key
        self.assertEqual(grant_key(["uv", "sync"]), "shell_exec:dev:uv:sync")
        self.assertEqual(grant_key(["ruff", "check", "src"]), "shell_exec:dev:ruff:check")
        self.assertEqual(grant_key(["python", "-m", "pytest", "tests"]), "shell_exec:dev:python:-m:pytest")
        self.assertEqual(grant_key(["python3", "scripts/build.py"]), "shell_exec:dev:python3:scripts/build.py")
        self.assertEqual(grant_key(["npx", "-y", "playwright", "test"]), "shell_exec:dev:npx:playwright")
