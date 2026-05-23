from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Optional

from app_core.logging import get_logger
from scripts.boss_detector.scanner import BossScanner
from scripts.boss_extractor.extractor import BossNameExtractor
from scripts.run_exporter.video_frame_provider import VideoFrameProvider

logger = get_logger(__name__)

# Matches filenames produced by RunExporter: {video_stem}_run_{index}.json
_RUN_JSON_RE = re.compile(r"^(.+)_run_(\d+)\.json$")


class BossProcessor:
    def __init__(
        self,
        frame_provider: VideoFrameProvider,
        boss_scanner: BossScanner,
        boss_name_extractor: Optional[BossNameExtractor] = None,
    ) -> None:
        self.frame_provider = frame_provider
        self.boss_scanner = boss_scanner
        self.boss_name_extractor = boss_name_extractor

    def _find_video(self, video_dir: Path, stem: str) -> Optional[Path]:
        """Find a video file whose stem matches the extracted video stem."""
        for ext in (".mp4", ".mkv", ".avi", ".mov"):
            candidate = video_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _process_run_json(self, run_json_path: Path, video_dir: Path) -> None:
        match = _RUN_JSON_RE.match(run_json_path.name)
        if not match:
            logger.warning(
                "Skipping %s: filename does not match expected pattern",
                run_json_path.name,
            )
            return

        video_stem = match.group(1)
        video_path = self._find_video(video_dir, video_stem)
        if video_path is None:
            logger.warning(
                "Skipping %s: no matching video found for stem %r",
                run_json_path.name,
                video_stem,
            )
            return

        with open(run_json_path, encoding="utf-8") as f:
            run_data: dict = json.load(f)

        start_time: Optional[float] = run_data.get("start_time")
        end_time: Optional[float] = run_data.get("end_time")

        if start_time is None or end_time is None:
            logger.warning(
                "Skipping %s: missing start_time or end_time", run_json_path.name
            )
            return

        fps = self.frame_provider.get_fps(video_path.name)
        logger.debug(
            "%s: fps=%.2f start=%.2fs end=%.2fs",
            run_json_path.name,
            fps,
            start_time,
            end_time,
        )

        try:
            segments = self.boss_scanner.scan_time_range(
                frame_provider=self.frame_provider,
                video_name=video_path.name,
                start_time=start_time,
                end_time=end_time,
                fps=fps,
                run_end_time=end_time,
            )
        except Exception as e:
            logger.error("Boss scan failed for %s: %s", run_json_path.name, e)
            return
        finally:
            self.frame_provider.release_video(video_path.name)
            gc.collect()

        logger.info("%s: %d boss segment(s) found", run_json_path.name, len(segments))

        boss_fights = []
        for seg in segments:
            if self.boss_name_extractor is not None:
                try:
                    fb = self.frame_provider.get_frame_bytes(
                        video_name=video_path.name,
                        frame_index=seg.sample_frame_index,
                    )
                    name_result = self.boss_name_extractor.extract_name(fb)
                    del fb
                    seg.boss_names = name_result.boss_names
                    logger.debug(
                        "  class=%s -> names=%r", seg.boss_class, seg.boss_names
                    )
                except Exception as e:
                    logger.warning(
                        "  boss name extraction failed for class=%s: %s",
                        seg.boss_class,
                        e,
                    )
                    seg.boss_names = []
                finally:
                    self.frame_provider.release_video(video_path.name)
                    gc.collect()

            if not seg.boss_names:
                fallback_name = (
                    seg.boss_class
                    if seg.boss_class
                    and seg.boss_class not in {"boss", "regular_gameplay"}
                    else "Unknown Boss"
                )
                seg.boss_names = [fallback_name]
                logger.debug(
                    "  segment [%.1fs-%.1fs] no boss name extracted, using fallback=%r",
                    seg.start_time,
                    seg.end_time,
                    fallback_name,
                )

            boss_fights.append(
                {
                    "boss_names": seg.boss_names,
                    "boss_class": seg.boss_class,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration_seconds": seg.duration,
                    "player_died": seg.player_died,
                }
            )

        run_data["boss_fights"] = boss_fights

        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        logger.info("%s: wrote %d boss fight(s)", run_json_path.name, len(boss_fights))

    def process_folder(self, run_json_dir: Path, video_dir: Path) -> None:
        run_jsons = sorted(run_json_dir.glob("*_run_*.json"))

        if not run_jsons:
            logger.warning("No run JSON files found in: %s", run_json_dir)
            return

        logger.info(
            "Boss processing: %d run JSON(s) in %s", len(run_jsons), run_json_dir
        )

        ok = 0
        failed = 0

        for idx, run_json_path in enumerate(run_jsons, start=1):
            logger.info("[%d/%d] %s", idx, len(run_jsons), run_json_path.name)
            try:
                self._process_run_json(run_json_path, video_dir)
                ok += 1
            except Exception as e:
                logger.error("FAILED: %s — %s", run_json_path.name, e)
                failed += 1
            finally:
                if self.boss_name_extractor is not None:
                    self.boss_name_extractor.reset_session()
                gc.collect()

        logger.info("Done. Successful: %d  Failed: %d", ok, failed)
