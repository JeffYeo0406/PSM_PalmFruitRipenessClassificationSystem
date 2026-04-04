from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parent


def resolve_path(path_value: str | None) -> Optional[str]:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p)


def latest_match(patterns: Iterable[str]) -> Optional[str]:
    candidates = []
    for pattern in patterns:
        candidates.extend(glob(str(ROOT_DIR / pattern)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_artifact(
    env_value: str | None,
    default_candidates: Iterable[str],
    glob_patterns: Iterable[str],
    allow_missing_default: bool = True,
) -> Optional[str]:
    env_path = resolve_path(env_value)
    if env_path and Path(env_path).exists():
        return env_path

    for candidate in default_candidates:
        candidate_path = resolve_path(candidate)
        if candidate_path and Path(candidate_path).exists():
            return candidate_path

    newest = latest_match(glob_patterns)
    if newest:
        return newest

    if allow_missing_default:
        first = next(iter(default_candidates), None)
        return resolve_path(first)

    return env_path
