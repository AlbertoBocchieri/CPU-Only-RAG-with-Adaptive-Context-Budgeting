from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil

_POWER_MW_RE = re.compile(r"^\s*([A-Za-z0-9 \-_]+Power):\s*([0-9]+(?:\.[0-9]+)?)\s*mW\b", re.IGNORECASE)
_POWERMETRICS_STATUS_LOCK = threading.Lock()
_POWERMETRICS_UNAVAILABLE_REASON: str | None = None


@dataclass(slots=True)
class ResourceStats:
    rss_peak_bytes: int
    rss_mean_bytes: float
    cpu_peak_pct: float
    cpu_mean_pct: float
    rss_timeseries: list[dict[str, float]]
    cpu_timeseries: list[dict[str, float]]
    power_peak_watts: float | None
    power_mean_watts: float | None
    power_samples: int
    power_status: str
    power_backend: str | None


class _PowermetricsSampler:
    def __init__(self, sampling_interval_ms: int = 1000) -> None:
        self.sampling_interval_ms = max(100, int(sampling_interval_ms))
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._running = False
        self._samples_watts: list[float] = []
        self._status = "disabled"
        self._backend = "powermetrics"
        self._lock = threading.Lock()

    def start(self) -> None:
        global _POWERMETRICS_UNAVAILABLE_REASON
        with _POWERMETRICS_STATUS_LOCK:
            unavailable_reason = _POWERMETRICS_UNAVAILABLE_REASON

        if unavailable_reason:
            self._status = unavailable_reason
            return

        if os.name != "posix":
            self._status = "unsupported_os"
            with _POWERMETRICS_STATUS_LOCK:
                _POWERMETRICS_UNAVAILABLE_REASON = self._status
            return
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            self._status = "no_permission"
            with _POWERMETRICS_STATUS_LOCK:
                _POWERMETRICS_UNAVAILABLE_REASON = self._status
            return
        if shutil.which("powermetrics") is None:
            self._status = "missing_binary"
            with _POWERMETRICS_STATUS_LOCK:
                _POWERMETRICS_UNAVAILABLE_REASON = self._status
            return

        cmd = [
            "powermetrics",
            "--samplers",
            "cpu_power",
            "-i",
            str(self.sampling_interval_ms),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._status = f"spawn_error:{exc.__class__.__name__}"
            return

        self._status = "starting"
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._running = False
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        with self._lock:
            samples = list(self._samples_watts)
            status = self._status

        if status == "starting":
            status = "ok" if samples else "no_samples"
        if status in {"unsupported_os", "missing_binary", "no_permission"}:
            with _POWERMETRICS_STATUS_LOCK:
                global _POWERMETRICS_UNAVAILABLE_REASON
                _POWERMETRICS_UNAVAILABLE_REASON = status

        power_peak = max(samples) if samples else None
        power_mean = float(sum(samples) / len(samples)) if samples else None
        return {
            "power_peak_watts": power_peak,
            "power_mean_watts": power_mean,
            "power_samples": len(samples),
            "power_status": status,
            "power_backend": self._backend,
        }

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            if not self._running:
                break
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            if "must be invoked as the superuser" in low:
                with self._lock:
                    self._status = "no_permission"
                break

            watts = self._extract_cpu_power_watts(line)
            if watts is not None:
                with self._lock:
                    self._samples_watts.append(watts)
                    self._status = "ok"

    @staticmethod
    def _extract_cpu_power_watts(line: str) -> float | None:
        match = _POWER_MW_RE.match(line)
        if not match:
            return None
        label = match.group(1).strip().lower().replace("-", " ")
        if label != "cpu power":
            return None
        try:
            milliwatts = float(match.group(2))
        except ValueError:
            return None
        return milliwatts / 1000.0


class QueryResourceSampler:
    def __init__(
        self,
        sampling_interval_ms: int = 200,
        include_timeseries: bool = False,
        timeseries_stride: int = 5,
        profile_power: bool = False,
        power_sampling_interval_ms: int = 1000,
    ) -> None:
        self.sampling_interval_ms = max(20, int(sampling_interval_ms))
        self.include_timeseries = bool(include_timeseries)
        self.timeseries_stride = max(1, int(timeseries_stride))
        self.profile_power = bool(profile_power)
        self.power_sampling_interval_ms = max(100, int(power_sampling_interval_ms))
        self.process = psutil.Process()

        self._running = False
        self._thread: threading.Thread | None = None
        self._power_sampler: _PowermetricsSampler | None = None
        self._start_ts = 0.0
        self._rss_values: list[int] = []
        self._cpu_values: list[float] = []
        self._rss_ts: list[dict[str, float]] = []
        self._cpu_ts: list[dict[str, float]] = []
        self._sample_idx = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_ts = time.perf_counter()
        self._rss_values.clear()
        self._cpu_values.clear()
        self._rss_ts.clear()
        self._cpu_ts.clear()
        self._sample_idx = 0

        # Prime cpu_percent so next calls measure interval deltas.
        self.process.cpu_percent(interval=None)

        self._power_sampler = None
        if self.profile_power:
            self._power_sampler = _PowermetricsSampler(
                sampling_interval_ms=self.power_sampling_interval_ms,
            )
            self._power_sampler.start()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceStats:
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        power_stats: dict[str, Any] = {
            "power_peak_watts": None,
            "power_mean_watts": None,
            "power_samples": 0,
            "power_status": ("disabled" if not self.profile_power else "unavailable"),
            "power_backend": ("powermetrics" if self.profile_power else None),
        }
        if self._power_sampler is not None:
            power_stats = self._power_sampler.stop()
        return self._build_stats(power_stats=power_stats)

    def _loop(self) -> None:
        interval_s = self.sampling_interval_ms / 1000.0
        while self._running:
            rss = int(self.process.memory_info().rss)
            cpu = float(self.process.cpu_percent(interval=None))
            self._rss_values.append(rss)
            self._cpu_values.append(cpu)

            if self.include_timeseries and (self._sample_idx % self.timeseries_stride == 0):
                t_ms = (time.perf_counter() - self._start_ts) * 1000.0
                self._rss_ts.append({"timestamp_ms": t_ms, "rss_bytes": float(rss)})
                self._cpu_ts.append({"timestamp_ms": t_ms, "cpu_pct": cpu})
            self._sample_idx += 1
            time.sleep(interval_s)

    def _build_stats(self, power_stats: dict[str, Any]) -> ResourceStats:
        rss = self._rss_values
        cpu = self._cpu_values
        rss_peak = max(rss) if rss else 0
        rss_mean = float(sum(rss) / len(rss)) if rss else 0.0
        cpu_peak = max(cpu) if cpu else 0.0
        cpu_mean = float(sum(cpu) / len(cpu)) if cpu else 0.0
        return ResourceStats(
            rss_peak_bytes=rss_peak,
            rss_mean_bytes=rss_mean,
            cpu_peak_pct=cpu_peak,
            cpu_mean_pct=cpu_mean,
            rss_timeseries=list(self._rss_ts),
            cpu_timeseries=list(self._cpu_ts),
            power_peak_watts=power_stats.get("power_peak_watts"),
            power_mean_watts=power_stats.get("power_mean_watts"),
            power_samples=int(power_stats.get("power_samples", 0)),
            power_status=str(power_stats.get("power_status", "unavailable")),
            power_backend=power_stats.get("power_backend"),
        )
