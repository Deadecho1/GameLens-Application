from __future__ import annotations

from .config import VERSIONS_ROOT
from .models import VersionInfo


class VersionRepository:
    def __init__(self) -> None:
        VERSIONS_ROOT.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> list[VersionInfo]:
        versions: list[VersionInfo] = []
        for path in sorted(VERSIONS_ROOT.iterdir()):
            if not path.is_dir():
                continue
            versions.append(self._build_version(path.name))
        return versions

    def ensure_version(self, version_name: str) -> VersionInfo:
        cleaned_name = version_name.strip()
        if not cleaned_name:
            raise ValueError("Version name cannot be empty.")

        version = self._build_version(cleaned_name)
        version.event_json_dir.mkdir(parents=True, exist_ok=True)
        version.run_json_dir.mkdir(parents=True, exist_ok=True)
        return version

    def _build_version(self, version_name: str) -> VersionInfo:
        root = VERSIONS_ROOT / version_name
        return VersionInfo(
            name=version_name,
            root_dir=root,
            event_json_dir=root / "event_json",
            run_json_dir=root / "run_json",
        )