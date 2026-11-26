# SPDX-License-Identifier: GPL-3.0-or-later
"""SOMA-77 in numpy (no torch, no Kimodo): rest data, rest-convention change, BVH read and write (MG8).

The link worker runs in the MotionBricks venv, which has no torch, so the two Kimodo steps it needs are ported here:
- `to_standard_tpose` / `from_standard_tpose` (Kimodo `kimodo/skeleton/transforms.py` `change_tpose` at commit e1c219c, Apache-2.0,
  github.com/nv-tlabs/kimodo): global rotations times the per-joint `global_rot_offsets` (transposed one way), then
  back to local rotations;
- `motion_to_bvh` (`kimodo/exports/bvh.py`): wrapper joint `Root` (6 channels, zero), Hips (6 channels, absolute cm,
  dummy OFFSET 0 100 0), the rest ZYX rotation, cm, no End Site blocks.
Rest data (`soma77_rest.json`: joint names, parents, both rest poses' joint positions, the offsets) was extracted once
from Kimodo's `somaskel77` skeleton assets. Arrays: local rotations (T, 77, 3, 3) in Kimodo joint order, root
(= Hips) positions (T, 3) in m, Y up.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from .clipstats import _rot, parse_bvh

HERE = Path(__file__).resolve().parent
REST_FILE = HERE / "soma77_rest.json"
HIPS_DUMMY_OFFSET_CM = (0.0, 100.0, 0.0)  # Kimodo's nominal pelvis OFFSET (format, not a tunable)
REST_TOL_CM = 1e-3 * 100  # read_hub_take's 1e-3 m offset tolerance


@functools.lru_cache(maxsize=1)
def rest() -> dict:
    d = json.loads(REST_FILE.read_text(encoding="utf-8"))
    parents = d["parents"]
    assert all(p < i for i, p in enumerate(parents)), "SOMA-77 joints must be in topological order"
    u, _, vt = np.linalg.svd(np.array(d["global_rot_offsets"]))  # Kimodo stores float32: re-orthonormalize (~1e-7)
    return {"names": d["names"], "parents": parents, "index": {n: i for i, n in enumerate(d["names"])},
            "tpose": np.array(d["tpose_joints"]), "seed": np.array(d["seed_joints"]), "offsets": u @ vt}


def bone_offsets(kind: str) -> np.ndarray:
    """(77, 3) local joint offsets (m) of rest `kind` ('tpose' or 'seed'); the root's is its position (zero)."""
    r = rest()
    j = r[kind]
    par = np.array([max(p, 0) for p in r["parents"]])
    off = j - j[par]
    off[0] = j[0]
    return off


# ---------------------------------------------------------------- rotations


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions x y z w -> (..., 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) quaternions x y z w (w >= 0), Shepperd's method."""
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4))
    for i, r in enumerate(flat):
        t = np.trace(r)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
        q = np.array(q)
        out[i] = -q if q[3] < 0 else q
    return (out / np.linalg.norm(out, axis=-1, keepdims=True)).reshape(m.shape[:-2] + (4,))


def mat_to_euler_zyx_deg(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) degrees (z, y, x) with M = Rz(z) Ry(y) Rx(x): the BVH `Zrotation Yrotation Xrotation`."""
    s = np.clip(-m[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(s)
    gimbal = np.abs(s) > 1 - 1e-9
    z = np.where(gimbal, 0.0, np.arctan2(m[..., 1, 0], m[..., 0, 0]))
    x = np.where(gimbal, np.arctan2(m[..., 2, 0] * -m[..., 0, 1], m[..., 1, 1]), np.arctan2(m[..., 2, 1], m[..., 2, 2]))
    return np.degrees(np.stack([z, y, x], -1))


def rot_y(theta: float) -> np.ndarray:
    """Rotation about +Y by `theta` radians (+Z turns toward +X)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between rotation matrices (..., 3, 3)."""
    c = (np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------- rest convention


def globals_from_locals(local: np.ndarray) -> np.ndarray:
    g = np.empty_like(local)
    for i, p in enumerate(rest()["parents"]):
        g[:, i] = local[:, i] if p < 0 else g[:, p] @ local[:, i]
    return g


def locals_from_globals(g: np.ndarray) -> np.ndarray:
    loc = np.empty_like(g)
    for i, p in enumerate(rest()["parents"]):
        loc[:, i] = g[:, i] if p < 0 else np.swapaxes(g[:, p], -1, -2) @ g[:, i]
    return loc


def to_standard_tpose(local_seed: np.ndarray) -> np.ndarray:
    """BONES-SEED rest locals -> standard T-pose locals (Kimodo `to_standard_tpose`: G' = G @ offsets^T)."""
    off = rest()["offsets"]
    return locals_from_globals(globals_from_locals(local_seed) @ np.swapaxes(off, -1, -2)[None])


def from_standard_tpose(local_tpose: np.ndarray) -> np.ndarray:
    """Standard T-pose locals -> BONES-SEED rest locals (Kimodo `from_standard_tpose`: G' = G @ offsets)."""
    return locals_from_globals(globals_from_locals(local_tpose) @ rest()["offsets"][None])


def world_positions(local_tpose: np.ndarray, root: np.ndarray) -> np.ndarray:
    """(T, 77, 3) world joint positions (m) of T-pose-convention locals."""
    off = bone_offsets("tpose")
    g = globals_from_locals(local_tpose)
    pos = np.empty(local_tpose.shape[:2] + (3,))
    for i, p in enumerate(rest()["parents"]):
        pos[:, i] = root if p < 0 else pos[:, p] + np.einsum("tij,j->ti", g[:, p], off[i])
    return pos


# ---------------------------------------------------------------- BVH


@functools.lru_cache(maxsize=32)
def _read_cached(path: str, mtime_ns: int) -> tuple[np.ndarray, np.ndarray, str, float]:
    b = parse_bvh(Path(path))
    r = rest()
    names = [n for n in b.names if n != "Root"]
    if names != r["names"]:
        raise ValueError(f"{path}: not a SOMA-77 BVH in Kimodo joint order (a motiongen / Kimodo export)")
    T = b.motion.shape[0]
    local = np.zeros((T, len(b.names), 3, 3))
    hips_pos = None
    col = 0
    for j, chans in enumerate(b.channels):
        m = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        for k, ch in enumerate(chans):
            if ch.endswith("rotation"):
                m = m @ _rot(ch[0], b.motion[:, col + k])
        if b.names[j] == "Hips":
            hips_pos = np.zeros((T, 3))
            for k, ch in enumerate(chans):
                if ch.endswith("position"):
                    hips_pos[:, "XYZ".index(ch[0])] = b.motion[:, col + k]
        local[:, j] = m
        col += len(chans)
    if "Root" in b.names:  # Kimodo's wrapper: identity in its exports, folded into Hips if not
        ri, hi = b.names.index("Root"), b.names.index("Hips")
        local[:, hi] = local[:, ri] @ local[:, hi]
        local = np.delete(local, ri, axis=1)
    offs = np.array([o for n, o in zip(b.names, b.offsets) if n != "Root"])
    err = {k: float(np.abs(offs[1:] - bone_offsets(k)[1:] * 100).max()) for k in ("tpose", "seed")}
    kind = min(err, key=err.get)
    if err[kind] > REST_TOL_CM:
        raise ValueError(f"{path}: joint offsets match neither SOMA-77 rest pose (max error cm {err})")
    if kind == "seed":
        local = to_standard_tpose(local)
    return local, hips_pos / 100.0, kind, 1.0 / b.frame_time


def read_soma_bvh(path: Path) -> tuple[np.ndarray, np.ndarray, str, float]:
    """A SOMA-77 BVH (T-pose or BONES-SEED rest, detected from the offsets) -> (T-pose locals (T, 77, 3, 3),
    root positions (T, 3) m, rest kind, fps). numpy twin of `kimodo_backend.read_hub_take`."""
    p = Path(path)
    return _read_cached(str(p), p.stat().st_mtime_ns)


def _num(v: float) -> str:
    return repr(round(float(v), 6) + 0.0)


def bvh_text(local_tpose: np.ndarray, root: np.ndarray, fps: float, rest_kind: str = "tpose") -> str:
    """Kimodo-format SOMA-77 BVH text of T-pose-convention locals; `rest_kind` 'seed' writes the BONES-SEED rest."""
    r = rest()
    names, parents = r["names"], r["parents"]
    local = local_tpose if rest_kind == "tpose" else from_standard_tpose(local_tpose)
    off = bone_offsets(rest_kind) * 100
    children: dict[int, list[int]] = {i: [] for i in range(len(names))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    six = "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
    lines = ["HIERARCHY", "ROOT Root", "{", "  OFFSET 0.0 0.0 0.0", f"  {six}"]

    def joint(i: int, depth: int) -> None:
        pad = "  " * depth
        o = HIPS_DUMMY_OFFSET_CM if parents[i] < 0 else off[i]
        lines.extend([f"{pad}JOINT {names[i]}", f"{pad}{{", f"{pad}  OFFSET {' '.join(_num(v) for v in o)}",
                      f"{pad}  {six if parents[i] < 0 else 'CHANNELS 3 Zrotation Yrotation Xrotation'}"])
        for c in children[i]:
            joint(c, depth + 1)
        lines.append(f"{pad}}}")

    joint(0, 1)
    lines.append("}")
    T = len(local)
    eul = mat_to_euler_zyx_deg(local)  # (T, 77, 3)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps!r}"]
    order = []

    def walk(i: int) -> None:
        order.append(i)
        for c in children[i]:
            walk(c)

    walk(0)
    for t in range(T):
        vals = [0.0] * 6 + list(root[t] * 100)
        for i in order:
            vals.extend(eul[t, i])
        lines.append(" ".join(_num(v) for v in vals))
    return "\n".join(lines) + "\n"


def write_bvh(path: Path, local_tpose: np.ndarray, root: np.ndarray, fps: float, rest_kind: str = "tpose") -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(bvh_text(local_tpose, root, fps, rest_kind), encoding="utf-8")
    tmp.replace(path)
