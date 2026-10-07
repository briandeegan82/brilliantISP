#!/usr/bin/env python3
"""
Extract profiling results from terminal output in real-time.
"""

import re
import sys
from pathlib import Path


def extract_results(log_file: Path):
    """Extract profiling results from log file."""

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    content = log_file.read_text()

    # Find algorithm being profiled
    current_algorithm = None
    results = {}

    lines = content.split("\n")
    for i, line in enumerate(lines):
        # Check for algorithm being profiled
        if "Profiling demosaic algorithm:" in line:
            current_algorithm = line.split(":")[-1].strip()
            results[current_algorithm] = {"runs": [], "status": "in_progress"}
        elif "Profiling tone mapper:" in line:
            current_algorithm = line.split(":")[-1].strip()
            results[current_algorithm] = {"runs": [], "status": "in_progress"}

        # Check for run completion
        if current_algorithm and "run" in line and "=" in line and "s" in line:
            match = re.search(r"run\s+(\d+)/(\d+)\s+=\s+([\d.]+)s", line)
            if match:
                run_num, total_runs, time_sec = match.groups()
                results[current_algorithm]["runs"].append(float(time_sec))

        # Check for mean time
        if current_algorithm and "Mean:" in line:
            match = re.search(r"Mean:\s+([\d.]+)s", line)
            if match:
                results[current_algorithm]["mean"] = float(match.group(1))
                results[current_algorithm]["status"] = "completed"

        # Check for errors
        if current_algorithm and "ERROR:" in line:
            results[current_algorithm]["status"] = "error"
            results[current_algorithm]["error"] = line.split("ERROR:")[-1].strip()

    # Display results
    print("\n" + "=" * 70)
    print("PROFILING RESULTS (Real-time)")
    print("=" * 70 + "\n")

    if not results:
        print("No results found yet. Profiling may still be starting up...\n")
        return

    # Separate demosaic and tone mappers
    demosaic_results = {
        k: v
        for k, v in results.items()
        if k
        in [
            "bilinear",
            "malvar",
            "vng",
            "vng_opt",
            "hamilton_adams",
            "hamilton_adams_opt",
            "ppg",
            "ppg_opt",
            "lmmse",
            "lmmse_opt",
            "lmmse_fast",
            "ahd",
            "ahd_opt",
        ]
    }

    tmo_results = {
        k: v
        for k, v in results.items()
        if k in ["durand", "aces", "reinhard_integer", "aces_integer", "hable", "hable_integer"]
    }

    # Display demosaic results
    if demosaic_results:
        print("DEMOSAIC ALGORITHMS")
        print("-" * 70)
        print(f"{'Algorithm':<25} {'Status':<15} {'Mean Time':<15} {'Runs'}")
        print("-" * 70)

        for algo, data in sorted(demosaic_results.items()):
            status = data.get("status", "unknown")
            mean_time = f"{data.get('mean', 0):.3f}s" if "mean" in data else "computing..."
            runs = f"{len(data.get('runs', []))}/3" if "runs" in data else "0/3"

            if status == "error":
                mean_time = "ERROR"
                runs = data.get("error", "Unknown error")[:30]

            print(f"{algo:<25} {status:<15} {mean_time:<15} {runs}")
        print()

    # Display tone mapper results
    if tmo_results:
        print("TONE MAPPING OPERATORS")
        print("-" * 70)
        print(f"{'Tone Mapper':<25} {'Status':<15} {'Mean Time':<15} {'Runs'}")
        print("-" * 70)

        for tmo, data in sorted(tmo_results.items()):
            status = data.get("status", "unknown")
            mean_time = f"{data.get('mean', 0):.3f}s" if "mean" in data else "computing..."
            runs = f"{len(data.get('runs', []))}/3" if "runs" in data else "0/3"

            if status == "error":
                mean_time = "ERROR"
                runs = data.get("error", "Unknown error")[:30]

            print(f"{tmo:<25} {status:<15} {mean_time:<15} {runs}")
        print()

    # Summary
    completed = sum(1 for v in results.values() if v.get("status") == "completed")
    in_progress = sum(1 for v in results.values() if v.get("status") == "in_progress")
    errors = sum(1 for v in results.values() if v.get("status") == "error")
    total = len(results)

    print(f"Progress: {completed} completed, {in_progress} in progress, {errors} errors, {total} total")
    print()


if __name__ == "__main__":
    log_file = Path("/home/brian/.cursor/projects/home-brian-ISP-ws-brilliantISP/terminals/907431.txt")
    extract_results(log_file)
