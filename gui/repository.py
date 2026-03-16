from __future__ import annotations

from pathlib import Path

from .models import GameInfo, VersionInfo


class GameRepository:
    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir
        root_dir.mkdir(parents=True, exist_ok=True)

    def list_games(self) -> list[GameInfo]:
        games: list[GameInfo] = []
        for path in sorted(self._root.iterdir()):
            if path.is_dir():
                games.append(GameInfo(name=path.name, root_dir=path))
        return games

    def ensure_game(self, game_name: str) -> GameInfo:
        cleaned = game_name.strip()
        if not cleaned:
            raise ValueError("Game name cannot be empty.")
        game = GameInfo(name=cleaned, root_dir=self._root / cleaned)
        (game.root_dir / "versions").mkdir(parents=True, exist_ok=True)
        return game

    def list_versions(self, game: GameInfo) -> list[VersionInfo]:
        versions_dir = game.root_dir / "versions"
        if not versions_dir.exists():
            return []
        versions: list[VersionInfo] = []
        for path in sorted(versions_dir.iterdir()):
            if path.is_dir():
                versions.append(self._build_version(path))
        return versions

    def ensure_version(self, game: GameInfo, version_name: str) -> VersionInfo:
        cleaned = version_name.strip()
        if not cleaned:
            raise ValueError("Version name cannot be empty.")
        version_path = game.root_dir / "versions" / cleaned
        version = self._build_version(version_path)
        version.event_json_dir.mkdir(parents=True, exist_ok=True)
        version.run_json_dir.mkdir(parents=True, exist_ok=True)
        return version

    def _build_version(self, version_path: Path) -> VersionInfo:
        return VersionInfo(
            name=version_path.name,
            root_dir=version_path,
            event_json_dir=version_path / "event_json",
            run_json_dir=version_path / "run_json",
        )
