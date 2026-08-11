"""Tests for video frame extraction helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from video_extract import (  # noqa: E402
    extract_frames_from_video,
    list_image_frames,
    resolve_pano_input,
)


def test_list_image_frames(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("nope")
    frames = list_image_frames(tmp_path)
    assert [p.name for p in frames] == ["a.jpg", "b.png"]


def test_extract_frames_calls_ffmpeg(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "frames"

    def fake_run(cmd, check):
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_000001.jpg").write_bytes(b"jpg")
        (out / "frame_000002.jpg").write_bytes(b"jpg")
        return mock.Mock(returncode=0)

    with mock.patch("video_extract.ensure_ffmpeg", return_value="ffmpeg"), mock.patch(
        "video_extract.subprocess.run", side_effect=fake_run
    ) as run:
        frames = extract_frames_from_video(video, out, fps=2.0, overwrite=True)

    assert len(frames) == 2
    assert run.called
    cmd = run.call_args[0][0]
    assert "ffmpeg" in cmd[0]
    assert "fps=2.0" in cmd or "fps=2" in " ".join(cmd)
    meta = json.loads((out.parent / f"{out.name}_extract_meta.json").read_text())
    assert meta["n_frames"] == 2
    assert meta["fps"] == 2.0
    assert meta["max_frames"] is None


def test_extract_respects_max_frames(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "frames"

    def fake_run(cmd, check):
        assert "-frames:v" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "3"
        out.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            (out / f"frame_{i:06d}.jpg").write_bytes(b"jpg")
        return mock.Mock(returncode=0)

    with mock.patch("video_extract.ensure_ffmpeg", return_value="ffmpeg"), mock.patch(
        "video_extract.subprocess.run", side_effect=fake_run
    ):
        frames = extract_frames_from_video(video, out, fps=2.0, max_frames=3, overwrite=True)

    assert len(frames) == 3
    meta = json.loads((out.parent / f"{out.name}_extract_meta.json").read_text())
    assert meta["max_frames"] == 3


def test_max_frames_must_be_positive(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(ValueError):
        extract_frames_from_video(video, tmp_path / "out", fps=2.0, max_frames=0)


def test_extract_reuses_existing_without_overwrite(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "frames"
    out.mkdir()
    (out / "frame_000001.jpg").write_bytes(b"jpg")

    with mock.patch("video_extract.subprocess.run") as run:
        frames = extract_frames_from_video(video, out, fps=2.0, overwrite=False)
    assert len(frames) == 1
    run.assert_not_called()


def test_resolve_pano_input_dir(tmp_path: Path):
    d = tmp_path / "panos"
    d.mkdir()
    assert resolve_pano_input(pano_dir=d) == d.resolve()


def test_resolve_pano_input_rejects_both(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_pano_input(pano_dir=tmp_path, video=tmp_path / "v.mp4")


def test_fps_must_be_positive(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(ValueError):
        extract_frames_from_video(video, tmp_path / "out", fps=0)
