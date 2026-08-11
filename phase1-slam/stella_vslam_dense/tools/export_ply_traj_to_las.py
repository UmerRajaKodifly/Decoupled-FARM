#!/usr/bin/env python3
"""Export stella dense out.ply + TUM trajectories to LAS files."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError as e:  # pragma: no cover
    raise SystemExit("laspy is required: pip install laspy") from e


def parse_ply_header(f) -> Tuple[dict, int]:
    """Read PLY header; file cursor left at start of payload. Returns (info, header_bytes)."""
    first = f.readline()
    if not first:
        raise ValueError("Empty PLY")
    if isinstance(first, bytes):
        line0 = first.decode("ascii", errors="replace").strip()
        binary = True
    else:
        line0 = first.strip()
        binary = False
    if line0 != "ply":
        raise ValueError(f"Not a PLY file (got {line0!r})")

    fmt = None
    n_vertex = 0
    props: List[Tuple[str, str]] = []  # (type, name)
    header_lines = [line0]
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("Truncated PLY header")
        line = raw.decode("ascii", errors="replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        header_lines.append(line)
        if line.startswith("format "):
            parts = line.split()
            fmt = parts[1]  # ascii | binary_little_endian | binary_big_endian
        elif line.startswith("element vertex "):
            n_vertex = int(line.split()[-1])
        elif line.startswith("element ") and not line.startswith("element vertex "):
            # other elements not supported for streaming body of vertices-only files
            pass
        elif line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] != "list":
                props.append((parts[1], parts[2]))
        elif line == "end_header":
            break

    info = {
        "format": fmt,
        "n_vertex": n_vertex,
        "properties": props,
        "binary_mode": binary or (fmt is not None and fmt.startswith("binary")),
    }
    return info, 0


_PLY_TYPE_STRUCT = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "float32": "f",
    "float64": "d",
}


def _prop_indices(props: List[Tuple[str, str]]) -> Dict[str, int]:
    return {name: i for i, (_, name) in enumerate(props)}


def read_ply_xyz_rgb(path: Path, chunk_rows: int = 500_000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load XYZ (float64) and optional RGB uint8 (N,3) from PLY (ascii or binary)."""
    path = Path(path)
    # Open as binary first to detect and read header bytes accurately
    with open(path, "rb") as fb:
        header_bytes = b""
        while True:
            line = fb.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header_text = header_bytes.decode("ascii", errors="replace")
        info_lines = [ln.strip() for ln in header_text.splitlines()]
        fmt = None
        n_vertex = 0
        props: List[Tuple[str, str]] = []
        for line in info_lines[1:]:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property ") and "list" not in line.split()[1:2]:
                parts = line.split()
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))
        payload_offset = fb.tell()

    if n_vertex == 0:
        return np.zeros((0, 3), dtype=np.float64), None

    pidx = _prop_indices(props)
    for need in ("x", "y", "z"):
        if need not in pidx:
            raise ValueError(f"PLY missing property {need}: {path}")
    has_rgb = all(c in pidx for c in ("red", "green", "blue"))
    rgb_names = ("red", "green", "blue") if has_rgb else None
    # alternate common names
    if not has_rgb and all(c in pidx for c in ("r", "g", "b")):
        has_rgb = True
        rgb_names = ("r", "g", "b")

    xyz = np.empty((n_vertex, 3), dtype=np.float64)
    rgb = np.empty((n_vertex, 3), dtype=np.uint8) if has_rgb else None

    if fmt == "ascii":
        # Stream-parse ascii to limit peak temp memory from str arrays
        filled = 0
        with open(path, "r", encoding="ascii", errors="replace") as fa:
            for line in fa:
                if line.strip() == "end_header":
                    break
            # Vectorized chunks via line read + fromstring
            buf: List[str] = []
            xi, yi, zi = pidx["x"], pidx["y"], pidx["z"]
            if has_rgb:
                assert rgb_names is not None
                ri, gi, bi = pidx[rgb_names[0]], pidx[rgb_names[1]], pidx[rgb_names[2]]

            def flush(lines: List[str], start: int) -> int:
                if not lines:
                    return start
                # each line: space-separated fields
                arr = np.loadtxt(lines, dtype=np.float64, ndmin=2)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                n = arr.shape[0]
                end = start + n
                xyz[start:end, 0] = arr[:, xi]
                xyz[start:end, 1] = arr[:, yi]
                xyz[start:end, 2] = arr[:, zi]
                if rgb is not None:
                    rgb[start:end, 0] = np.clip(arr[:, ri], 0, 255).astype(np.uint8)
                    rgb[start:end, 1] = np.clip(arr[:, gi], 0, 255).astype(np.uint8)
                    rgb[start:end, 2] = np.clip(arr[:, bi], 0, 255).astype(np.uint8)
                return end

            for line in fa:
                line = line.strip()
                if not line:
                    continue
                buf.append(line)
                if len(buf) >= chunk_rows:
                    filled = flush(buf, filled)
                    buf = []
            filled = flush(buf, filled)
        if filled != n_vertex:
            # Allow shorter (truncated) but warn via return size
            xyz = xyz[:filled]
            if rgb is not None:
                rgb = rgb[:filled]
        return xyz, rgb

    # binary
    endian = "<" if fmt == "binary_little_endian" else ">"
    fmt_chars = []
    for typ, name in props:
        sc = _PLY_TYPE_STRUCT.get(typ)
        if sc is None:
            raise ValueError(f"Unsupported PLY type {typ} for property {name}")
        fmt_chars.append(sc)
    rec_fmt = endian + "".join(fmt_chars)
    rec_size = struct.calcsize(rec_fmt)
    xi, yi, zi = pidx["x"], pidx["y"], pidx["z"]
    if has_rgb:
        assert rgb_names is not None
        ri, gi, bi = pidx[rgb_names[0]], pidx[rgb_names[1]], pidx[rgb_names[2]]

    with open(path, "rb") as fb:
        fb.seek(payload_offset)
        start = 0
        while start < n_vertex:
            n = min(chunk_rows, n_vertex - start)
            raw = fb.read(rec_size * n)
            if len(raw) < rec_size * n:
                n = len(raw) // rec_size
                if n == 0:
                    break
                raw = raw[: n * rec_size]
            # unpack via numpy
            dt = np.dtype([(f"f{i}", endian + fc) for i, fc in enumerate(fmt_chars)])
            rec = np.frombuffer(raw, dtype=dt, count=n)
            fields = [rec[f"f{i}"] for i in range(len(fmt_chars))]
            xyz[start : start + n, 0] = fields[xi]
            xyz[start : start + n, 1] = fields[yi]
            xyz[start : start + n, 2] = fields[zi]
            if rgb is not None:
                rgb[start : start + n, 0] = np.asarray(fields[ri], dtype=np.uint8)
                rgb[start : start + n, 1] = np.asarray(fields[gi], dtype=np.uint8)
                rgb[start : start + n, 2] = np.asarray(fields[bi], dtype=np.uint8)
            start += n
        if start != n_vertex:
            xyz = xyz[:start]
            if rgb is not None:
                rgb = rgb[:start]
    return xyz, rgb


def write_las_xyz_rgb(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> int:
    """Write LAS 1.2 point format 2 (XYZ+RGB) or format 0 if no RGB. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    if n == 0:
        # still write empty-ish file? write empty LAS with 0 points
        header = laspy.LasHeader(point_format=2 if rgb is not None else 0, version="1.2")
        las = laspy.LasData(header)
        las.write(str(path))
        return 0

    # scale/offset for lossless float32-ish accuracy
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    scale = 0.0001  # 0.1 mm
    # expand range slightly
    point_format = 2 if rgb is not None else 0
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.offsets = mins.tolist()
    header.scales = [scale, scale, scale]
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    if rgb is not None:
        # LAS stores 16-bit color; expand 8-bit as value << 8 | value
        r = rgb[:, 0].astype(np.uint16)
        g = rgb[:, 1].astype(np.uint16)
        b = rgb[:, 2].astype(np.uint16)
        las.red = (r << 8) | r
        las.green = (g << 8) | g
        las.blue = (b << 8) | b
    las.write(str(path))
    return n


def load_tum_positions(traj_path: Path) -> np.ndarray:
    """Load TUM trajectory; return (N,3) translations."""
    rows = []
    with open(traj_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            rows.append((tx, ty, tz))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    """Unit-ish sphere samples on sphere of given radius (n points)."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - (i / (n - 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    pts = np.stack([x, y, z], axis=1) * radius
    return pts


def trajectory_to_spheres(
    positions: np.ndarray,
    radius: float = 0.1,
    samples_per_sphere: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sphere point cloud with red→green color by progress."""
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    sphere = fibonacci_sphere(samples_per_sphere, radius)
    xyz_list = []
    rgb_list = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        r = int(round(255 * (1.0 - t)))
        g = int(round(255 * t))
        b = 0
        pts = sphere + positions[i]
        xyz_list.append(pts)
        col = np.tile(np.array([r, g, b], dtype=np.uint8), (samples_per_sphere, 1))
        rgb_list.append(col)
    return np.vstack(xyz_list), np.vstack(rgb_list)


def resolve_traj_file(run_dir: Path) -> Optional[Path]:
    frame = run_dir / "traj" / "frame_trajectory.txt"
    key = run_dir / "traj" / "keyframe_trajectory.txt"
    for p in (frame, key):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def export_one(
    name: str,
    rohtas_root: Path,
    out_dir: Path,
    sphere_radius: float = 0.1,
    sphere_samples: int = 64,
) -> dict:
    run_dir = rohtas_root / name
    ply_path = run_dir / "out.ply"
    result = {
        "name": name,
        "ok_ply": False,
        "ok_traj": False,
        "ply_points": 0,
        "traj_poses": 0,
        "traj_points": 0,
        "ply_las": None,
        "traj_las": None,
        "error": None,
    }
    try:
        if not ply_path.is_file():
            raise FileNotFoundError(f"Missing PLY: {ply_path}")
        print(f"[{name}] Reading PLY {ply_path} ...", flush=True)
        xyz, rgb = read_ply_xyz_rgb(ply_path)
        las_path = out_dir / f"{name}.las"
        print(f"[{name}] Writing LAS ({xyz.shape[0]} pts) -> {las_path}", flush=True)
        n_pts = write_las_xyz_rgb(las_path, xyz, rgb)
        result["ok_ply"] = True
        result["ply_points"] = n_pts
        result["ply_las"] = str(las_path)

        traj_path = resolve_traj_file(run_dir)
        if traj_path is None:
            print(f"[{name}] No trajectory file; skipping sphere LAS", flush=True)
            result["error"] = "no trajectory"
            return result
        print(f"[{name}] Trajectory from {traj_path.name}", flush=True)
        pos = load_tum_positions(traj_path)
        result["traj_poses"] = int(pos.shape[0])
        s_xyz, s_rgb = trajectory_to_spheres(pos, radius=sphere_radius, samples_per_sphere=sphere_samples)
        traj_las = out_dir / "trajectories" / f"{name}_trajectory.las"
        print(f"[{name}] Writing traj LAS ({s_xyz.shape[0]} pts, {pos.shape[0]} poses) -> {traj_las}", flush=True)
        n_s = write_las_xyz_rgb(traj_las, s_xyz, s_rgb)
        result["ok_traj"] = True
        result["traj_points"] = n_s
        result["traj_las"] = str(traj_las)
        # free memory sooner for large clouds
        del xyz, rgb, s_xyz, s_rgb, pos
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[{name}] FAILED: {result['error']}", file=sys.stderr, flush=True)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export stella out.ply + trajectories to LAS")
    ap.add_argument(
        "--rohtas-root",
        type=Path,
        required=True,
        help="Root with per-video folders containing out.ply and traj/",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for *.las and trajectories/*_trajectory.las",
    )
    ap.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Video/folder names to process",
    )
    ap.add_argument("--sphere-radius", type=float, default=0.1, help="Trajectory sphere radius (m)")
    ap.add_argument("--sphere-samples", type=int, default=64, help="Points per pose sphere")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.names:
        results.append(
            export_one(
                name,
                args.rohtas_root,
                out_dir,
                sphere_radius=args.sphere_radius,
                sphere_samples=args.sphere_samples,
            )
        )

    # chmod readable
    try:
        import os
        import stat

        for root, dirs, files in os.walk(out_dir):
            os.chmod(root, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            for fn in files:
                fp = Path(root) / fn
                os.chmod(fp, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass

    print("\n=== Summary ===", flush=True)
    failed = 0
    for r in results:
        ply_sz = Path(r["ply_las"]).stat().st_size if r["ply_las"] and Path(r["ply_las"]).is_file() else 0
        traj_sz = Path(r["traj_las"]).stat().st_size if r["traj_las"] and Path(r["traj_las"]).is_file() else 0
        status = "OK" if r["ok_ply"] and r["ok_traj"] else "PARTIAL/FAIL"
        if not (r["ok_ply"] and r["ok_traj"]):
            failed += 1
        print(
            f"{r['name']}: {status} | cloud_pts={r['ply_points']} cloud_las={ply_sz}B "
            f"| poses={r['traj_poses']} traj_pts={r['traj_points']} traj_las={traj_sz}B "
            f"| err={r['error']}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
