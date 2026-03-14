from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import os


@dataclass(frozen=True)
class ProjectPaths:
    script_dir: Path
    project_root: Path
    raw_dir: Path
    output_dir: Path
    stage_dir: Path
    task_dir: Path


def _walk_up(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    for parent in current.parents:
        yield parent


def _looks_like_project_root(path: Path) -> bool:
    path = path.resolve()
    strong_markers = [
        path / 'Raw Data',
        path / 'outputs_capstone',
        path / '.git',
        path / 'requirements.txt',
        path / 'README.md',
    ]
    if any(marker.exists() for marker in strong_markers):
        return True

    if (path / 'src').exists() and ((path / 'Raw Data').exists() or (path / 'outputs_capstone').exists()):
        return True

    return False


def find_project_root(start: Optional[Path] = None, project_root: Optional[str | Path] = None) -> Path:
    explicit_root = project_root or os.getenv('CAPSTONE_PROJECT_ROOT')
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    start_path = Path(start or Path.cwd()).resolve()
    for candidate in _walk_up(start_path):
        if _looks_like_project_root(candidate):
            return candidate

    if start_path.is_file():
        start_path = start_path.parent

    if start_path.name.lower() == 'src' and start_path.parent.exists():
        return start_path.parent.resolve()

    for parent in start_path.parents:
        if parent.name.lower() == 'src' and parent.parent.exists():
            return parent.parent.resolve()

    return start_path.resolve()


def build_project_paths(
    script_file: Path,
    project_root: Optional[str | Path] = None,
    raw_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
) -> ProjectPaths:
    script_path = Path(script_file).resolve()
    script_dir = script_path.parent
    root = find_project_root(script_path, project_root=project_root)

    raw_value = raw_dir or os.getenv('CAPSTONE_RAW_DIR')
    out_value = out_dir or os.getenv('CAPSTONE_OUTPUT_DIR')

    resolved_raw_dir = Path(raw_value).expanduser().resolve() if raw_value else (root / 'Raw Data' / 'RQ1').resolve()
    resolved_output_dir = Path(out_value).expanduser().resolve() if out_value else (root / 'outputs_capstone').resolve()

    return ProjectPaths(
        script_dir=script_dir,
        project_root=root,
        raw_dir=resolved_raw_dir,
        output_dir=resolved_output_dir,
        stage_dir=(resolved_output_dir / 'stage').resolve(),
        task_dir=(resolved_output_dir / 'tasks').resolve(),
    )


def candidate_task_dirs(paths: ProjectPaths, extra_dirs: Optional[Sequence[str | Path]] = None) -> list[Path]:
    script_parent = paths.script_dir.parent if paths.script_dir.parent.exists() else paths.script_dir
    base_candidates = [
        paths.task_dir,
        paths.output_dir,
        paths.script_dir / 'tasks',
        paths.script_dir / 'outputs_capstone' / 'tasks',
        paths.script_dir / 'outputs_capstone',
        script_parent / 'outputs_capstone' / 'tasks',
        script_parent / 'outputs_capstone',
        script_parent / 'tasks',
        paths.script_dir,
        script_parent,
        paths.project_root / 'tasks',
        paths.project_root / 'outputs_capstone' / 'tasks',
        paths.project_root / 'outputs_capstone',
        paths.project_root,
    ]
    if extra_dirs:
        base_candidates.extend(Path(p).expanduser().resolve() for p in extra_dirs)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in base_candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def resolve_task_file(filename: str, paths: ProjectPaths, extra_dirs: Optional[Sequence[str | Path]] = None) -> Path:
    directories = candidate_task_dirs(paths, extra_dirs=extra_dirs)
    for directory in directories:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    searched = '\n'.join(f'- {directory}' for directory in directories)
    raise FileNotFoundError(
        f'Could not find {filename}. Looked in:\n{searched}\n\n'
        'Set CAPSTONE_PROJECT_ROOT, CAPSTONE_OUTPUT_DIR, or pass --project_root/--out_dir if needed.'
    )


def resolve_optional_task_file(
    filenames: Sequence[str],
    paths: ProjectPaths,
    extra_dirs: Optional[Sequence[str | Path]] = None,
) -> Optional[Path]:
    for filename in filenames:
        try:
            return resolve_task_file(filename, paths, extra_dirs=extra_dirs)
        except FileNotFoundError:
            continue
    return None
