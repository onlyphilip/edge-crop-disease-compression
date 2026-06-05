"""Practical edge deployment metrics for Keras and TFLite inference."""

from __future__ import annotations

from collections.abc import Callable
import cProfile
from dataclasses import dataclass
from pathlib import Path
import platform
import pstats
import resource
import statistics
import time
from typing import Any


NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class EdgeModelSpec:
    """Description of a model variant to benchmark."""

    model_name: str
    backend: str
    model_path: Path
    variant: str
    purpose: str
    available: bool = True


def optional_import(module_name: str) -> Any | None:
    """Import an optional dependency without failing the benchmark."""
    try:
        return __import__(module_name)
    except Exception:
        return None


def model_size_mb(model_path: str | Path) -> float | None:
    """Return model file size in MiB when the model exists."""
    path = Path(model_path).expanduser()
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size / (1024.0 * 1024.0)


def array_nbytes(value: Any) -> int | str:
    """Return byte size for numpy-like tensors."""
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return int(value.__sizeof__())
    except Exception:
        return NOT_AVAILABLE


def percentile(values: list[float], q: float) -> float | None:
    """Compute percentile without requiring numpy."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def get_process_memory_mb() -> float | str:
    """Return current process RSS memory in MiB when psutil is available."""
    psutil = optional_import("psutil")
    if psutil is not None:
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system().lower() == "darwin":
            return float(peak) / (1024.0 * 1024.0)
        return float(peak) / 1024.0
    except Exception:
        return NOT_AVAILABLE


def get_process_cpu_percent() -> float | str:
    """Return current process CPU percent when psutil is available."""
    psutil = optional_import("psutil")
    if psutil is None:
        return NOT_AVAILABLE
    try:
        return float(psutil.Process().cpu_percent(interval=None))
    except Exception:
        return NOT_AVAILABLE


def estimate_peak_memory_mb(inference_fn: Callable[[], Any]) -> float | str:
    """Measure peak memory with memory_profiler when installed."""
    memory_profiler = optional_import("memory_profiler")
    if memory_profiler is None:
        return NOT_AVAILABLE
    try:
        samples = memory_profiler.memory_usage((inference_fn, (), {}), interval=0.01, timeout=None)
        if not samples:
            return NOT_AVAILABLE
        return float(max(samples))
    except Exception:
        return NOT_AVAILABLE


def _start_codecarbon_tracker(output_dir: Path) -> Any | None:
    """Start a CodeCarbon tracker if the optional dependency works locally."""
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            output_dir=str(output_dir),
            output_file="codecarbon_edge_metrics.csv",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        return tracker
    except Exception:
        return None


def _stop_codecarbon_tracker(tracker: Any | None) -> tuple[float | str, float | str]:
    """Stop CodeCarbon and return energy/emissions when available."""
    if tracker is None:
        return NOT_AVAILABLE, NOT_AVAILABLE
    try:
        emissions = tracker.stop()
        energy = getattr(tracker, "final_emissions_data", None)
        energy_kwh = getattr(energy, "energy_consumed", NOT_AVAILABLE)
        return (
            float(energy_kwh) if energy_kwh != NOT_AVAILABLE and energy_kwh is not None else NOT_AVAILABLE,
            float(emissions) if emissions is not None else NOT_AVAILABLE,
        )
    except Exception:
        return NOT_AVAILABLE, NOT_AVAILABLE


def get_optional_hardware_energy_status() -> dict[str, str]:
    """Report optional hardware energy backends without requiring them."""
    pyrapl = optional_import("pyRAPL")
    pynvml = optional_import("pynvml")
    return {
        "pyrapl": "available" if pyrapl is not None and platform.system().lower() == "linux" else NOT_AVAILABLE,
        "pynvml": "available" if pynvml is not None else NOT_AVAILABLE,
    }


def profile_inference_once(inference_fn: Callable[[], Any], output_path: str | Path) -> str:
    """Write a small cProfile report for one inference call."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    inference_fn()
    profiler.disable()
    with destination.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats("cumtime")
        stats.print_stats(50)
    return str(destination)


def benchmark_inference(
    *,
    model_name: str,
    backend: str,
    model_path: str | Path,
    inference_fn: Callable[[], Any],
    warmup_runs: int,
    benchmark_runs: int,
    input_bytes: int | str,
    output_bytes_fn: Callable[[Any], int | str] = array_nbytes,
    energy_output_dir: str | Path | None = None,
    measure_peak_memory: bool = True,
    measure_energy: bool = True,
    profile_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure latency, memory, CPU, energy, communication, and stability."""
    path = Path(model_path).expanduser()
    warmup_runs = max(0, int(warmup_runs))
    benchmark_runs = max(1, int(benchmark_runs))

    raw: dict[str, Any] = {
        "model_name": model_name,
        "backend": backend,
        "model_path": str(path),
        "model_size_mb": model_size_mb(path),
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "input_bytes": input_bytes,
        "output_bytes": NOT_AVAILABLE,
        "latencies_ms": [],
        "status": "ok",
        "error_message": None,
        "successful_runs": 0,
        "failed_runs": 0,
        "profile_report": NOT_AVAILABLE,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "optional_energy_backends": get_optional_hardware_energy_status(),
        },
    }

    tracker = None
    try:
        for _ in range(warmup_runs):
            inference_fn()

        ram_before = get_process_memory_mb()
        _ = get_process_cpu_percent()
        if measure_energy:
            tracker = _start_codecarbon_tracker(Path(energy_output_dir or "."))

        output_sample = None
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            try:
                output_sample = inference_fn()
                raw["latencies_ms"].append((time.perf_counter() - start) * 1000.0)
                raw["successful_runs"] += 1
            except Exception as exc:
                raw["failed_runs"] += 1
                raw["status"] = "partial_failure"
                raw["error_message"] = str(exc)

        energy_kwh, emissions_kg = _stop_codecarbon_tracker(tracker)
        tracker = None
        cpu_after = get_process_cpu_percent()
        ram_after = get_process_memory_mb()

        raw["ram_before_mb"] = ram_before
        raw["ram_after_mb"] = ram_after
        raw["ram_delta_mb"] = (
            float(ram_after) - float(ram_before)
            if isinstance(ram_before, (int, float)) and isinstance(ram_after, (int, float))
            else NOT_AVAILABLE
        )
        raw["cpu_percent_avg"] = cpu_after
        raw["energy_kwh"] = energy_kwh
        raw["emissions_kg"] = emissions_kg
        raw["output_bytes"] = output_bytes_fn(output_sample) if output_sample is not None else NOT_AVAILABLE

        if measure_peak_memory:
            raw["peak_memory_mb"] = estimate_peak_memory_mb(inference_fn)
        else:
            raw["peak_memory_mb"] = NOT_AVAILABLE

        if profile_output_path is not None:
            raw["profile_report"] = profile_inference_once(inference_fn, profile_output_path)

        latencies = raw["latencies_ms"]
        if latencies:
            avg = float(statistics.fmean(latencies))
            raw["avg_latency_ms"] = avg
            raw["p50_latency_ms"] = percentile(latencies, 50)
            raw["p95_latency_ms"] = percentile(latencies, 95)
            raw["min_latency_ms"] = float(min(latencies))
            raw["max_latency_ms"] = float(max(latencies))
            raw["latency_jitter_ms"] = float(statistics.pstdev(latencies)) if len(latencies) > 1 else 0.0
            raw["fps"] = float(1000.0 / avg) if avg > 0 else 0.0
        else:
            raw["status"] = "failed"
            raw["avg_latency_ms"] = None
            raw["p50_latency_ms"] = None
            raw["p95_latency_ms"] = None
            raw["min_latency_ms"] = None
            raw["max_latency_ms"] = None
            raw["latency_jitter_ms"] = None
            raw["fps"] = None
    except Exception as exc:
        _stop_codecarbon_tracker(tracker)
        raw.update(
            {
                "status": "failed",
                "error_message": str(exc),
                "ram_before_mb": NOT_AVAILABLE,
                "ram_after_mb": NOT_AVAILABLE,
                "ram_delta_mb": NOT_AVAILABLE,
                "peak_memory_mb": NOT_AVAILABLE,
                "cpu_percent_avg": NOT_AVAILABLE,
                "energy_kwh": NOT_AVAILABLE,
                "emissions_kg": NOT_AVAILABLE,
                "avg_latency_ms": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "fps": None,
            }
        )

    return raw
