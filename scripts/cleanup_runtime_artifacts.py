import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _build_patterns(*, keep_db: bool, all_pyc: bool) -> list[str]:
    patterns: list[str] = [
        "reports/api_smoke.jpg",
        "__pycache__/inference_db*.pyc",
        "scripts/__pycache__/pi_inference*.pyc",
        "scripts/__pycache__/predeploy_dry_run*.pyc",
        "scripts/__pycache__/smoke_check_inference_db*.pyc",
        "api/__pycache__/app*.pyc",
    ]

    if not keep_db:
        patterns.extend(
            [
                "reports/inference_log.db",
                "reports/inference_log.db-shm",
                "reports/inference_log.db-wal",
            ]
        )

    if all_pyc:
        patterns.append("**/__pycache__/*.pyc")

    return patterns


def _expand_targets(patterns: list[str]) -> list[Path]:
    targets: set[Path] = set()
    for pattern in patterns:
        targets.update(PROJECT_ROOT.glob(pattern))
    return sorted(path for path in targets if path.is_file())


def _prune_empty_cache_dirs(*, dry_run: bool) -> int:
    removed = 0
    cache_dirs = sorted(
        (path for path in PROJECT_ROOT.rglob("__pycache__") if path.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for cache_dir in cache_dirs:
        if any(cache_dir.iterdir()):
            continue
        print(f"[dir] remove {cache_dir.relative_to(PROJECT_ROOT)}")
        if not dry_run:
            cache_dir.rmdir()
        removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove generated runtime artifacts from dry runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed without deleting files",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep reports/inference_log.db and its sidecar WAL files",
    )
    parser.add_argument(
        "--all-pyc",
        action="store_true",
        help="Remove all .pyc files under any __pycache__ directory in the project",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = _build_patterns(keep_db=args.keep_db, all_pyc=args.all_pyc)
    targets = _expand_targets(patterns)

    if not targets:
        print("No matching generated artifacts found.")
        return

    file_removed = 0
    for path in targets:
        print(f"[file] remove {path.relative_to(PROJECT_ROOT)}")
        if not args.dry_run:
            path.unlink(missing_ok=True)
        file_removed += 1

    dir_removed = _prune_empty_cache_dirs(dry_run=args.dry_run)

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"cleanup_runtime_artifacts: {mode}")
    print(f"files={file_removed} dirs={dir_removed}")


if __name__ == "__main__":
    main()