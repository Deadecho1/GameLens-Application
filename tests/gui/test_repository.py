"""Unit tests for GameRepository — uses tmp_path, no hardcoded paths."""
from __future__ import annotations

import pytest

from gui.repository import GameRepository
from gui.models import GameInfo, VersionInfo


@pytest.fixture
def repo(tmp_path):
    return GameRepository(root_dir=tmp_path)


class TestListGames:
    def test_empty_when_no_games(self, repo):
        assert repo.list_games() == []

    def test_lists_game_after_creation(self, repo, tmp_path):
        (tmp_path / "MyGame").mkdir()
        games = repo.list_games()
        assert len(games) == 1
        assert games[0].name == "MyGame"

    def test_ignores_files(self, repo, tmp_path):
        (tmp_path / "not_a_game.txt").write_text("x")
        assert repo.list_games() == []

    def test_returns_sorted_order(self, repo, tmp_path):
        (tmp_path / "Zelda").mkdir()
        (tmp_path / "Elden Ring").mkdir()
        names = [g.name for g in repo.list_games()]
        assert names == sorted(names)


class TestEnsureGame:
    def test_creates_game_directory(self, repo, tmp_path):
        game = repo.ensure_game("TestGame")
        assert (tmp_path / "TestGame").is_dir()
        assert (tmp_path / "TestGame" / "versions").is_dir()

    def test_returns_game_info(self, repo):
        game = repo.ensure_game("TestGame")
        assert isinstance(game, GameInfo)
        assert game.name == "TestGame"

    def test_empty_name_raises(self, repo):
        with pytest.raises(ValueError):
            repo.ensure_game("  ")

    def test_idempotent_on_existing_dir(self, repo):
        repo.ensure_game("TestGame")
        game = repo.ensure_game("TestGame")  # should not raise
        assert game.name == "TestGame"


class TestListVersions:
    def test_no_versions_dir_returns_empty(self, repo):
        game = GameInfo(name="FakeGame", root_dir=repo._root / "FakeGame")
        assert repo.list_versions(game) == []

    def test_lists_version_after_creation(self, repo):
        game = repo.ensure_game("MyGame")
        repo.ensure_version(game, "v1.0")
        versions = repo.list_versions(game)
        assert len(versions) == 1
        assert versions[0].name == "v1.0"

    def test_version_has_correct_subdirs(self, repo):
        game = repo.ensure_game("MyGame")
        version = repo.ensure_version(game, "v1.0")
        assert version.event_json_dir.name == "event_json"
        assert version.run_json_dir.name == "run_json"


class TestEnsureVersion:
    def test_creates_subdirectories(self, repo):
        game = repo.ensure_game("MyGame")
        version = repo.ensure_version(game, "v1.0")
        assert version.event_json_dir.is_dir()
        assert version.run_json_dir.is_dir()

    def test_empty_name_raises(self, repo):
        game = repo.ensure_game("MyGame")
        with pytest.raises(ValueError):
            repo.ensure_version(game, "")
