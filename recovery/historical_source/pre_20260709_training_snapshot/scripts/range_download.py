from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mb", type=int, default=128)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = _probe_total_size(args.url, args.user_agent)
    done_path = args.out.with_suffix(args.out.suffix + ".segments.json")
    done = _load_done(done_path)
    existing_size = args.out.stat().st_size if args.out.exists() else 0

    chunk = int(args.chunk_mb) * 1024 * 1024
    ranges = []
    for start in range(0, total, chunk):
        end = min(start + chunk - 1, total - 1)
        key = f"{start}-{end}"
        if key in done:
            continue
        if end < existing_size:
            done[key] = True
            continue
        ranges.append((start, end))
    _save_done(done_path, done)

    print(f"url={args.url}")
    print(f"out={args.out}")
    print(f"total={total} bytes ({total / 1024**3:.2f} GiB)")
    print(f"existing={existing_size} bytes ({existing_size / 1024**3:.2f} GiB)")
    print(f"pending_segments={len(ranges)} workers={args.workers} chunk_mb={args.chunk_mb}")

    if not args.out.exists():
        args.out.write_bytes(b"")

    lock = threading.Lock()
    completed_bytes = sum(
        _segment_size(key)
        for key in done
    )
    start_time = time.time()

    def mark_done(start: int, end: int) -> None:
        nonlocal completed_bytes
        key = f"{start}-{end}"
        with lock:
            done[key] = True
            completed_bytes += end - start + 1
            _save_done(done_path, done)
            elapsed = max(time.time() - start_time, 1e-6)
            pct = 100.0 * completed_bytes / total
            speed = completed_bytes / elapsed / 1024**2
            print(f"done {key}  {pct:.2f}%  avg={speed:.2f} MiB/s", flush=True)

    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = [
            pool.submit(
                _download_range,
                args.url,
                args.out,
                start,
                end,
                args.user_agent,
                int(args.retries),
            )
            for start, end in ranges
        ]
        for future in as_completed(futures):
            start, end = future.result()
            mark_done(start, end)

    final_done = _load_done(done_path)
    expected = {f"{start}-{min(start + chunk - 1, total - 1)}" for start in range(0, total, chunk)}
    missing = sorted(expected - set(final_done))
    if missing:
        raise SystemExit(f"Missing {len(missing)} segments; rerun the same command to resume.")
    actual = args.out.stat().st_size
    if actual != total:
        raise SystemExit(f"Size mismatch: expected {total}, got {actual}")
    print("download complete")


def _probe_total_size(url: str, user_agent: str) -> int:
    headers = {"User-Agent": user_agent, "Range": "bytes=0-0"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)\s*$", content_range)
        if match:
            return int(match.group(1))
        length = response.headers.get("Content-Length")
        if length:
            return int(length)
    raise RuntimeError("Could not determine remote file size")


def _download_range(
    url: str,
    out: Path,
    start: int,
    end: int,
    user_agent: str,
    retries: int,
) -> tuple[int, int]:
    headers = {"User-Agent": user_agent, "Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Expected 206, got {response.status_code}")
                written = 0
                with out.open("r+b") as f:
                    f.seek(start)
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        f.write(block)
                        written += len(block)
                if written != expected:
                    raise RuntimeError(f"Short range {start}-{end}: {written} != {expected}")
                return start, end
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("unreachable")


def _load_done(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_done(path: Path, done: dict[str, bool]) -> None:
    path.write_text(json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")


def _segment_size(key: str) -> int:
    start_text, end_text = key.split("-", 1)
    return int(end_text) - int(start_text) + 1


if __name__ == "__main__":
    main()
