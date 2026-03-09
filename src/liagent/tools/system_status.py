"""Read-only local system status tool."""

from __future__ import annotations

import os
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

from . import ToolCapability, tool

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

_MAX_OUTPUT_CHARS = 1200


def _format_gib(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "unavailable"
    gib = float(num_bytes) / (1024 ** 3)
    return f"{gib:.1f} GiB"


def _process_rss_bytes() -> int:
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss_raw)
    return int(rss_raw) * 1024


def _cpu_snapshot() -> tuple[int, float | None, tuple[float, float, float] | None]:
    cores = os.cpu_count() or 0
    cpu_percent = None
    if psutil is not None:
        try:
            cpu_percent = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            cpu_percent = None
    load_avg = None
    try:
        la = os.getloadavg()
        load_avg = (float(la[0]), float(la[1]), float(la[2]))
    except (AttributeError, OSError):
        load_avg = None
    return cores, cpu_percent, load_avg


def _linux_memory_snapshot() -> tuple[int | None, int | None, int | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None, None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable") or values.get("MemFree")
    if total is None or available is None:
        return None, None, None, None
    used = max(total - available, 0)
    percent = (used / total * 100.0) if total else None
    return total, available, used, percent


def _mac_memory_snapshot() -> tuple[int | None, int | None, int | None, float | None]:
    try:
        total_raw = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        total = int(total_raw.stdout.strip())
    except Exception:
        return None, None, None, None

    try:
        vm_stat = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except Exception:
        return total, None, None, None

    page_size = 4096
    pages: dict[str, int] = {}
    for line in vm_stat.stdout.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 4096
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip().rstrip(".")
        try:
            pages[key.strip()] = int(raw_value)
        except ValueError:
            continue

    free_pages = pages.get("Pages free", 0)
    inactive_pages = pages.get("Pages inactive", 0)
    speculative_pages = pages.get("Pages speculative", 0)
    available = (free_pages + inactive_pages + speculative_pages) * page_size
    used = max(total - available, 0)
    percent = (used / total * 100.0) if total else None
    return total, available, used, percent


def _memory_snapshot() -> tuple[int | None, int | None, int | None, float | None]:
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available), int(vm.used), float(vm.percent)
        except Exception:
            pass
    system = platform.system()
    if system == "Linux":
        return _linux_memory_snapshot()
    if system == "Darwin":
        return _mac_memory_snapshot()
    return None, None, None, None


def _temperature_snapshot() -> str:
    if psutil is not None and hasattr(psutil, "sensors_temperatures"):
        try:
            sensors = psutil.sensors_temperatures() or {}
            readings = []
            for entries in sensors.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if current is not None:
                        readings.append(float(current))
            if readings:
                avg = sum(readings) / len(readings)
                return f"{avg:.1f}°C"
        except Exception:
            pass
    return "unavailable"


@tool(
    name="system_status",
    description="Read local machine CPU, memory, disk, and LiAgent process RSS without executing arbitrary code.",
    risk_level="low",
    requires_confirmation=False,
    capability=ToolCapability(
        data_classification="internal",
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="fast",
        failure_modes=("probe_unavailable",),
    ),
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)
async def system_status(**_kwargs) -> str:
    system = platform.system() or "unknown"
    release = platform.release() or "unknown"
    cores, cpu_percent, load_avg = _cpu_snapshot()
    mem_total, mem_available, mem_used, mem_percent = _memory_snapshot()
    disk = shutil.disk_usage(Path.home())
    rss_bytes = _process_rss_bytes()
    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"platform: {system} {release}",
    ]

    if cpu_percent is not None:
        cpu_line = f"cpu: {cpu_percent:.1f}% across {cores} cores"
    elif load_avg is not None:
        cpu_line = (
            f"cpu: load avg {load_avg[0]:.2f}/{load_avg[1]:.2f}/{load_avg[2]:.2f}"
            + (f" across {cores} cores" if cores else "")
        )
    else:
        cpu_line = f"cpu: unavailable" + (f" ({cores} cores detected)" if cores else "")
    lines.append(cpu_line)

    if mem_total is not None and mem_used is not None and mem_percent is not None:
        lines.append(
            f"memory: {_format_gib(mem_used)} used / {_format_gib(mem_total)} total "
            f"({mem_percent:.0f}%); available {_format_gib(mem_available)}"
        )
    else:
        lines.append("memory: unavailable")

    disk_used = max(disk.used, 0)
    disk_pct = (disk_used / disk.total * 100.0) if disk.total else 0.0
    lines.append(
        f"disk ({Path.home()}): {_format_gib(disk_used)} used / {_format_gib(disk.total)} total "
        f"({disk_pct:.0f}%); free {_format_gib(disk.free)}"
    )
    lines.append(f"liagent_process_rss: {_format_gib(rss_bytes)}")
    lines.append(f"temperature: {_temperature_snapshot()}")
    return "\n".join(lines)
