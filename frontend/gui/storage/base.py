from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def get_items(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]: ...

    @abstractmethod
    def get_bosses(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]: ...

    @abstractmethod
    def get_runs(self, user_id: int, game_name: str, version_name: str | None) -> list[dict]: ...

    @abstractmethod
    def get_stats(self, user_id: int, game_name: str, version_name: str | None) -> dict: ...

    @abstractmethod
    def check_processed_videos(
        self, user_id: int, game_name: str, version_name: str | None, video_names: list[str]
    ) -> list[str]: ...
