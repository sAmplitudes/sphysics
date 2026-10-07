#!/usr/bin/env python3
"""
Benchmark a Python script: wall-clock runtime + memory (min/max/avg),
aggregated across the script and ALL of its threads and child processes.

Displays live statistics while the target runs.

Usage:
    python benchmark.py --runs 5 -- python my_script.py --arg1 foo
    python benchmark.py --runs 5 --warmup 1 --json out.json -- python my_script.py
    python benchmark.py --no-live -- python my_script.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field

import psutil

MB = 1024 * 1024


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """Metrics for a single execution of the target script."""
    wall_time_s: float
    cpu_time_s: float          # user+sys, self + all reaped descendants
    mem_min_mb: float
    mem_max_mb: float          # peak total RSS across the process tree
    mem_avg_mb: float          # time-weighted average
    mem_final_mb: float
    max_proc_count: int        # peak number of live processes in the tree
    max_thread_count: int      # peak number of live threads in the tree
    sample_count: int
    exit_code: int
    stdout: str = field(repr=False, default="")
    stderr: str = field(repr=False, default="")


@dataclass
class LiveSnapshot:
    """A consistent point-in-time view of the sampler's state."""
    elapsed_s: float
    current_mb: float
    min_mb: float
    max_mb: float
    avg_mb: float
    proc_count: int
    thread_count: int
    sample_count: int


# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------

class TreeMemorySampler(threading.Thread):
    """
    Samples aggregate RSS of a process and every descendant, on an interval.

    Design notes:
      * We aggregate RSS over the whole tree. Threads share their parent's
        address space, so their memory is already counted in the parent's RSS;
        threads are only counted separately for reporting.
      * RSS double-counts pages shared via copy-on-write after fork(). See
        the USS note in the docstring of `_take_snapshot` for the tradeoff.
      * We keep the interval tight and tolerate races: processes can die
        between enumeration and measurement, which raises NoSuchProcess.
      * Running aggregates are maintained incrementally so `live_snapshot()`
        is O(1) and cheap enough to call from a display thread.

    NOTE ON NAMING: the stop flag is `_stop_evt`, NOT `_stop`.
    threading.Thread has a private method named `_stop()` which
    `join()` calls internally on timeout. Shadowing it with an Event
    raises "TypeError: 'Event' object is not callable".
    """

    def __init__(self, root_pid: int, interval: float = 0.02, use_uss: bool = False):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self.use_uss = use_uss
        self._stop_evt = threading.Event()

        # Guards every mutable field below.
        self._lock = threading.Lock()

        self.samples: list[tuple[float, int]] = []   # (timestamp, bytes)
        self.max_procs = 0
        self.max_threads = 0

        # Running aggregates for cheap live reads.
        self._t_start: float | None = None
        self._cur_bytes = 0
        self._min_bytes: int | None = None
        self._max_bytes = 0
        self._cur_procs = 0
        self._cur_threads = 0
        self._area = 0.0                     # trapezoidal integral, byte-seconds
        self._last_point: tuple[float, int] | None = None

    # ---- main loop --------------------------------------------------------

    def run(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop_evt.is_set():
            self._record(self._take_snapshot(root))
            self._stop_evt.wait(self.interval)

        # One last sample so very short-lived scripts aren't left with zero data.
        self._record(self._take_snapshot(root))

    def _record(self, snap: tuple[int, int, int] | None) -> None:
        """Fold one raw reading into the sample list and running aggregates."""
        if snap is None:
            return
        total, nprocs, nthreads = snap
        now = time.perf_counter()

        with self._lock:
            if self._t_start is None:
                self._t_start = now

            self.samples.append((now, total))

            # Trapezoidal area accumulates incrementally, so the time-weighted
            # average never requires a second pass over the sample list.
            if self._last_point is not None:
                t0, v0 = self._last_point
                self._area += (v0 + total) / 2.0 * (now - t0)
            self._last_point = (now, total)

            self._cur_bytes = total
            self._max_bytes = max(self._max_bytes, total)
            self._min_bytes = total if self._min_bytes is None \
                else min(self._min_bytes, total)

            self._cur_procs = nprocs
            self._cur_threads = nthreads
            self.max_procs = max(self.max_procs, nprocs)
            self.max_threads = max(self.max_threads, nthreads)

    def _take_snapshot(self, root: psutil.Process) -> tuple[int, int, int] | None:
        """
        Walk the tree once and sum memory. Returns (bytes, n_procs, n_threads)
        or None if the root has already exited.

        RSS vs USS:
          RSS is cheap and universally available, but pages shared between
          parent and forked children are counted once per process. USS (unique
          set size) avoids that inflation and is the more honest number for
          fork-heavy workloads -- notably, NumPy-backed data structures share
          large read-only buffers across workers, so RSS wildly overstates
          real consumption there. USS is much slower to compute and needs
          elevated privileges on some platforms, hence opt-in.
        """
        total = 0
        nprocs = 0
        nthreads = 0

        try:
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return None

        for p in procs:
            try:
                if self.use_uss:
                    total += p.memory_full_info().uss
                else:
                    total += p.memory_info().rss
                nthreads += p.num_threads()
                nprocs += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process vanished or is not introspectable; skip it.
                continue

        if nprocs == 0:
            return None
        return total, nprocs, nthreads

    def stop(self) -> None:
        self._stop_evt.set()

    # ---- live read --------------------------------------------------------

    def live_snapshot(self) -> LiveSnapshot | None:
        """
        Thread-safe O(1) view of current state. Returns None before the
        first successful sample.
        """
        with self._lock:
            if self._t_start is None or self._min_bytes is None:
                return None

            span = self._last_point[0] - self._t_start if self._last_point else 0.0
            avg = (self._area / span) if span > 0 else float(self._cur_bytes)

            return LiveSnapshot(
                elapsed_s=time.perf_counter() - self._t_start,
                current_mb=self._cur_bytes / MB,
                min_mb=self._min_bytes / MB,
                max_mb=self._max_bytes / MB,
                avg_mb=avg / MB,
                proc_count=self._cur_procs,
                thread_count=self._cur_threads,
                sample_count=len(self.samples),
            )

    # ---- final statistics -------------------------------------------------

    def stats_mb(self) -> tuple[float, float, float, float]:
        """Return (min, max, time_weighted_avg, final) in MiB."""
        with self._lock:
            if not self.samples:
                return (0.0, 0.0, 0.0, 0.0)

            values = [b for _, b in self.samples]
            lo, hi, last = min(values), max(values), values[-1]

            # Time-weighted average. A plain mean would bias toward whatever
            # phase happened to produce the most samples.
            if len(self.samples) > 1:
                span = self.samples[-1][0] - self.samples[0][0]
                avg = self._area / span if span > 0 else float(statistics.fmean(values))
            else:
                avg = float(values[0])

            return (lo / MB, hi / MB, avg / MB, last / MB)


# --------------------------------------------------------------------------
# Live display
# --------------------------------------------------------------------------

class LiveDisplay(threading.Thread):
    """
    Renders periodic status updates from a sampler while the target runs.

    Two modes:
      inplace=True  -> single line rewritten with \\r (interactive terminals)
      inplace=False -> one appended line per update (logs, CI, pipes)

    Writes to stderr so stdout stays clean for piping.

    NOTE ON NAMING: see the same warning on TreeMemorySampler. The stop
    flag must not be called `_stop`.
    """

    SPINNER = "|/-\\"

    def __init__(
        self,
        sampler: TreeMemorySampler,
        refresh: float = 0.1,
        inplace: bool = True,
        label: str = "",
        stream=None,
    ):
        super().__init__(daemon=True)
        self.sampler = sampler
        self.refresh = refresh
        self.inplace = inplace
        self.label = label
        self.stream = stream or sys.stderr
        self._stop_evt = threading.Event()
        self._tick = 0
        self._last_width = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._render()
            self._stop_evt.wait(self.refresh)
        self._render(final=True)
        self._finish_line()

    def _render(self, final: bool = False) -> None:
        snap = self.sampler.live_snapshot()
        if snap is None:
            return

        if self.inplace:
            spin = "*" if final else self.SPINNER[self._tick % len(self.SPINNER)]
            self._tick += 1
            line = (
                f"{self.label}{spin} "
                f"{snap.elapsed_s:6.2f}s | "
                f"cur {snap.current_mb:8.1f} | "
                f"min {snap.min_mb:8.1f} | "
                f"avg {snap.avg_mb:8.1f} | "
                f"max {snap.max_mb:8.1f} MiB | "
                f"{snap.proc_count:2d}p/{snap.thread_count:3d}t"
            )
            # Truncate to terminal width so wrapping never breaks \r redraw.
            width = shutil.get_terminal_size((100, 24)).columns
            if len(line) > width - 1:
                line = line[: width - 1]
            # Pad over any longer previous line, then return the cursor.
            pad = max(0, self._last_width - len(line))
            self._last_width = len(line)
            self.stream.write("\r" + line + " " * pad + "\r" + line)
            self.stream.flush()
        else:
            self.stream.write(
                f"{self.label}[{snap.elapsed_s:7.2f}s] "
                f"cur={snap.current_mb:.1f} min={snap.min_mb:.1f} "
                f"avg={snap.avg_mb:.1f} max={snap.max_mb:.1f} MiB  "
                f"procs={snap.proc_count} threads={snap.thread_count} "
                f"samples={snap.sample_count}\n"
            )
            self.stream.flush()

    def _finish_line(self) -> None:
        if self.inplace and self._last_width:
            self.stream.write("\n")
            self.stream.flush()
            self._last_width = 0

    def stop(self) -> None:
        self._stop_evt.set()


# --------------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------------

def run_once(
    cmd: list[str],
    interval: float = 0.02,
    use_uss: bool = False,
    capture: bool = True,
    timeout: float | None = None,
    live: bool = False,
    live_refresh: float = 0.1,
    live_inplace: bool = True,
    live_label: str = "",
) -> RunResult:
    """Execute `cmd` once under measurement."""

    # Unbuffered child output avoids stdio buffering skewing short runs.
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    # start_new_session puts the child in its own process group so we can
    # clean up the entire tree reliably on timeout.
    popen_kwargs: dict = dict(env=env, start_new_session=True)
    if capture:
        popen_kwargs.update(
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, **popen_kwargs)

    sampler = TreeMemorySampler(proc.pid, interval=interval, use_uss=use_uss)
    sampler.start()

    display: LiveDisplay | None = None
    if live:
        display = LiveDisplay(
            sampler,
            refresh=live_refresh,
            inplace=live_inplace,
            label=live_label,
        )
        display.start()

    def shutdown_threads() -> None:
        """Stop sampler first so its final sample lands, then the display."""
        sampler.stop()
        sampler.join(timeout=2.0)
        if display is not None:
            display.stop()
            display.join(timeout=2.0)

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        shutdown_threads()
        _kill_tree(proc.pid)
        out, err = proc.communicate()
        raise
    else:
        shutdown_threads()
    finally:
        wall = time.perf_counter() - t0

    lo, hi, avg, final = sampler.stats_mb()

    # os.times() RUSAGE_CHILDREN semantics: covers this process's reaped
    # children AND their reaped descendants, so forked workers count.
    ru = os.times()
    cpu = ru.children_user + ru.children_system

    return RunResult(
        wall_time_s=wall,
        cpu_time_s=cpu,
        mem_min_mb=lo,
        mem_max_mb=hi,
        mem_avg_mb=avg,
        mem_final_mb=final,
        max_proc_count=sampler.max_procs,
        max_thread_count=sampler.max_threads,
        sample_count=len(sampler.samples),
        exit_code=proc.returncode,
        stdout=out or "" if capture else "",
        stderr=err or "" if capture else "",
    )


def _kill_tree(pid: int) -> None:
    """Terminate a process and all descendants, escalating to SIGKILL."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=3)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


# --------------------------------------------------------------------------
# Multi-run aggregation
# --------------------------------------------------------------------------

def aggregate(runs: list[RunResult]) -> dict:
    def summarize(vals: list[float]) -> dict:
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    return {
        "runs": len(runs),
        "wall_time_s": summarize([r.wall_time_s for r in runs]),
        "cpu_time_s": summarize([r.cpu_time_s for r in runs]),
        "mem_peak_mb": summarize([r.mem_max_mb for r in runs]),
        "mem_avg_mb": summarize([r.mem_avg_mb for r in runs]),
        "mem_min_mb": summarize([r.mem_min_mb for r in runs]),
        "max_proc_count": max(r.max_proc_count for r in runs),
        "max_thread_count": max(r.max_thread_count for r in runs),
        "exit_codes": sorted({r.exit_code for r in runs}),
    }


def print_report(agg: dict, per_run: list[RunResult]) -> None:
    def row(label: str, s: dict, unit: str) -> str:
        return (f"  {label:<16} min={s['min']:9.3f}  "
                f"avg={s['mean']:9.3f}  max={s['max']:9.3f}  "
                f"median={s['median']:9.3f}  sd={s['stdev']:7.3f} {unit}")

    print("\n" + "=" * 74)
    print(f"BENCHMARK RESULTS  ({agg['runs']} run(s))")
    print("=" * 74)
    print(row("wall time", agg["wall_time_s"], "s"))
    print(row("cpu time", agg["cpu_time_s"], "s"))
    print(row("peak memory", agg["mem_peak_mb"], "MiB"))
    print(row("avg memory", agg["mem_avg_mb"], "MiB"))
    print(row("min memory", agg["mem_min_mb"], "MiB"))
    print(f"\n  peak processes in tree : {agg['max_proc_count']}")
    print(f"  peak threads in tree   : {agg['max_thread_count']}")
    print(f"  exit codes             : {agg['exit_codes']}")

    if len(per_run) > 1:
        print("\n  per-run detail:")
        print(f"    {'#':>3} {'wall(s)':>9} {'peak(MiB)':>11} "
              f"{'avg(MiB)':>10} {'samples':>8} {'exit':>5}")
        for i, r in enumerate(per_run, 1):
            print(f"    {i:>3} {r.wall_time_s:9.3f} {r.mem_max_mb:11.2f} "
                  f"{r.mem_avg_mb:10.2f} {r.sample_count:8d} {r.exit_code:5d}")
    print("=" * 74 + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark runtime and memory of a command, "
                    "including all forked children.",
        epilog="Example: benchmark.py --runs 5 -- python train.py --epochs 3",
    )
    ap.add_argument("--runs", type=int, default=1,
                    help="measured repetitions (default: 1)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="unmeasured warmup runs, to prime caches (default: 0)")
    ap.add_argument("--interval", type=float, default=0.2,
                    help="memory sampling interval in seconds (default: 0.2)")
    ap.add_argument("--uss", action="store_true",
                    help="use USS instead of RSS (slower, avoids "
                         "double-counting shared pages after fork)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="kill the process tree after N seconds")
    ap.add_argument("--show-output", action="store_true",
                    help="stream the script's output instead of capturing it")
    ap.add_argument("--json", metavar="PATH",
                    help="write full results as JSON")

    live_grp = ap.add_argument_group("live display")
    live_grp.add_argument("--no-live", action="store_true",
                          help="disable live statistics during execution")
    live_grp.add_argument("--live-refresh", type=float, default=1.0,
                          help="live redraw interval in seconds (default: 1.0)")
    live_grp.add_argument("--live-lines", action="store_true",
                          help="force append-only live output instead of "
                               "in-place updates (useful for logs/CI)")
    live_grp.add_argument("--live-warmup", action="store_true",
                          help="also show live stats during warmup runs")

    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="command to benchmark (prefix with --)")
    args = ap.parse_args()

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given. Example: benchmark.py -- python my_script.py")

    live = not args.no_live

    # In-place \r updates require a TTY and exclusive use of the line. If the
    # child is streaming to the same terminal, its writes would shred the
    # redraw, so fall back to append-mode lines.
    inplace = (
        not args.live_lines
        and sys.stderr.isatty()
        and not args.show_output
    )

    print(f"Command : {' '.join(cmd)}")
    print(f"Metric  : {'USS' if args.uss else 'RSS'} "
          f"@ {args.interval * 1000:.0f} ms")
    if live:
        mode = "in-place" if inplace else "line-per-update"
        print(f"Live    : on ({mode}, {args.live_refresh * 1000:.0f} ms refresh)")
    print()

    for i in range(args.warmup):
        label = f"  warmup {i + 1}/{args.warmup}  "
        if live and args.live_warmup:
            print(f"Warmup {i + 1}/{args.warmup}")
        else:
            print(f"Warmup {i + 1}/{args.warmup} ...", flush=True)
        run_once(
            cmd, args.interval, args.uss,
            capture=not args.show_output, timeout=args.timeout,
            live=live and args.live_warmup,
            live_refresh=args.live_refresh,
            live_inplace=inplace,
            live_label=label,
        )

    results: list[RunResult] = []
    for i in range(args.runs):
        label = f"  run {i + 1}/{args.runs}  "
        if live:
            print(f"Run {i + 1}/{args.runs}")
        else:
            print(f"Run {i + 1}/{args.runs} ...", end=" ", flush=True)

        r = run_once(
            cmd, args.interval, args.uss,
            capture=not args.show_output, timeout=args.timeout,
            live=live,
            live_refresh=args.live_refresh,
            live_inplace=inplace,
            live_label=label,
        )

        summary = (f"{r.wall_time_s:.3f}s, peak {r.mem_max_mb:.1f} MiB, "
                   f"{r.max_proc_count} proc / {r.max_thread_count} thr")
        if live:
            print(f"  -> {summary}")
        else:
            print(summary)

        if r.exit_code != 0:
            print(f"  !! exit code {r.exit_code}", file=sys.stderr)
            if r.stderr:
                print("  stderr tail:", r.stderr.strip()[-800:], file=sys.stderr)
        results.append(r)

    agg = aggregate(results)
    print_report(agg, results)

    if args.json:
        payload = {
            "command": cmd,
            "config": {
                "runs": args.runs, "warmup": args.warmup,
                "interval_s": args.interval, "metric": "uss" if args.uss else "rss",
            },
            "aggregate": agg,
            "per_run": [
                {k: v for k, v in asdict(r).items()
                 if k not in ("stdout", "stderr")}
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON written to {args.json}")

    return 0 if all(r.exit_code == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
