from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CRITICAL_FILES = [
    "train.py",
    "src/friction_affordance/c3_experiment.py",
    "src/friction_affordance/models/c3_farnet.py",
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _config_chain(config_path: Path) -> list[Path]:
    chain: list[Path] = []
    seen: set[Path] = set()
    current = config_path.resolve()
    while current.exists() and current not in seen:
        seen.add(current)
        chain.append(current)
        parent: Path | None = None
        for line in current.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("extends:"):
                raw = stripped.split(":", 1)[1].strip().strip("'\"")
                parent = (current.parent / raw).resolve()
                break
        if parent is None:
            break
        current = parent
    return chain


def _record(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    stat = path.stat() if exists else None
    return {
        "path": _relative(path),
        "absolute_path": str(path),
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat else None,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
        "sha256": _sha256(path),
    }


def _target_files(configs: list[Path], extra_files: list[str]) -> list[Path]:
    targets = [_resolve(path) for path in DEFAULT_CRITICAL_FILES]
    for config in configs:
        targets.extend(_config_chain(config))
    for path in extra_files:
        targets.append(_resolve(path))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in targets:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare(manifest: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for frozen in manifest.get("files", []):
        current = _record(Path(str(frozen["absolute_path"])))
        ok = bool(current["exists"]) and current.get("sha256") == frozen.get("sha256")
        rows.append(
            {
                "path": frozen.get("path"),
                "absolute_path": frozen.get("absolute_path"),
                "pass": ok,
                "frozen_sha256": frozen.get("sha256"),
                "current_sha256": current.get("sha256"),
                "frozen_size_bytes": frozen.get("size_bytes"),
                "current_size_bytes": current.get("size_bytes"),
                "current_exists": current.get("exists"),
                "current_mtime": current.get("mtime"),
            }
        )
    return all(row["pass"] for row in rows), rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Candidate Integrity Verification")
    lines.append("")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Candidate: `{payload.get('candidate_name')}`")
    lines.append(f"- Overall pass: **{payload['ok']}**")
    lines.append(f"- Manifest: `{payload['manifest']}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Pass | Frozen SHA256 | Current SHA256 |")
    lines.append("|---|---:|---|---|")
    for row in payload["checks"]:
        frozen = row.get("frozen_sha256") or "-"
        current = row.get("current_sha256") or "-"
        if len(frozen) > 16:
            frozen = frozen[:16] + "..."
        if len(current) > 16:
            current = current[:16] + "..."
        lines.append(f"| `{row['path']}` | {row['pass']} | `{frozen}` | `{current}` |")
    lines.append("")
    if not payload["ok"]:
        lines.append("## Action")
        lines.append("")
        lines.append(
            "Do not launch the queued candidate under this manifest. Re-run readiness and create a new explicit freeze only after confirming the changed files are intentional."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify critical RSCD candidate files.")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["freeze", "check"], required=True)
    parser.add_argument("--config", action="append", default=[], type=Path)
    parser.add_argument("--file", action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "freeze":
        configs = [_resolve(config) for config in args.config]
        files = [_record(path) for path in _target_files(configs, args.file)]
        missing = [row for row in files if not row["exists"]]
        payload = {
            "mode": "freeze",
            "candidate_name": args.candidate_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(ROOT),
            "manifest": str(args.manifest),
            "ok": not missing,
            "files": files,
            "checks": [
                {
                    "path": row["path"],
                    "absolute_path": row["absolute_path"],
                    "pass": bool(row["exists"]),
                    "frozen_sha256": row["sha256"],
                    "current_sha256": row["sha256"],
                    "frozen_size_bytes": row["size_bytes"],
                    "current_size_bytes": row["size_bytes"],
                    "current_exists": row["exists"],
                    "current_mtime": row["mtime"],
                }
                for row in files
            ],
        }
        args.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        manifest = _load_manifest(args.manifest)
        ok, checks = _compare(manifest)
        payload = {
            "mode": "check",
            "candidate_name": args.candidate_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(ROOT),
            "manifest": str(args.manifest),
            "ok": ok,
            "checks": checks,
            "frozen_at": manifest.get("created_at"),
        }

    json_path = args.output_dir / "candidate_integrity_verification.json"
    md_path = args.output_dir / "candidate_integrity_verification.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(json.dumps({"ok": payload["ok"], "report": str(md_path)}, ensure_ascii=False))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
