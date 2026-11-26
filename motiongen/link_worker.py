# SPDX-License-Identifier: GPL-3.0-or-later
"""`python -m motiongen link <recipes...> --out <dir>` (MG8): MotionBricks link pieces between given poses.

One process, one resident MotionBricks SOMA engine (TensorRT plan cached in `--trt-cache`). Runs in the MotionBricks
venv (Python 3.12, no torch): pose sources are read and the output written by `motiongen.soma` (numpy port of
Kimodo's rest conversion and BVH export). Per take: `<out>/<id>/<id>.<nn>.bvh` (SOMA-77, standard T-pose rest, like
every motiongen take) + `.prov.json` + `.contacts.json` (MotionBricks foot contacts), one `jobs.jsonl` line.
Resumable: a take whose BVH and provenance exist is skipped.

Per job: context = the 4 source frames ending on `from`, target = `target_poses` frames starting at `to`, placed
next to the context (README "link"); then one raw generation (endpoint correction off: its endpoint misses are the
correction size), and two corrected ones (the take, and a repeat that measures determinism).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from . import TOOL_VERSION, soma
from .hubcheck import BODY_JOINTS
from .links import (DEFAULT_MB_ROOT, MB_FPS, POSES_PER_TOKEN, TRIM_FRAMES, LinkJob, PoseRef, link_plan_line,
                    load_links, plan_links)

MODEL_NAME = "MotionBricks-SOMA-RP-v1"


# ---------------------------------------------------------------- poses (numpy, no GPU)


class Sources:
    """Pose source BVHs, read once: T-pose locals + root (m)."""

    def __init__(self, root: Path | None):
        self.root = root

    def clip(self, ref: PoseRef) -> tuple[np.ndarray, np.ndarray, Path]:
        path = ref.resolve(self.root)
        if not path.exists():
            raise FileNotFoundError(f"{ref.label()}: {path} not found (relative paths resolve against --hub-root)")
        local, root, _rest, fps = soma.read_soma_bvh(path)
        if abs(fps - MB_FPS) > 1e-3:
            raise ValueError(f"{path}: {fps:g} fps; MotionBricks runs at {MB_FPS} Hz")
        return local, root, path

    @staticmethod
    def frame_of(ref: PoseRef, n: int) -> int:
        k = 0 if ref.at == "start" else n - 1 if ref.at == "end" else int(ref.at)
        if not 0 <= k < n:
            raise ValueError(f"{ref.label()}: frame {k} outside the take ({n} frames)")
        return k


def _feet(local: np.ndarray, root: np.ndarray) -> tuple[float, np.ndarray]:
    """(heading rad, mean ankle XZ) of one pose; heading from the ankles-to-toes vector, 0 = +Z, + toward +X."""
    w = soma.world_positions(local[None], root[None])[0]
    ix = soma.rest()["index"]
    f = w[ix["LeftToeBase"]] + w[ix["RightToeBase"]] - w[ix["LeftFoot"]] - w[ix["RightFoot"]]
    return math.atan2(f[0], f[2]), 0.5 * (w[ix["LeftFoot"]] + w[ix["RightFoot"]])[[0, 2]]


def _place(local: np.ndarray, root: np.ndarray, yaw: float, pivot_xz, t_xz) -> tuple[np.ndarray, np.ndarray]:
    """Turn the poses by `yaw` about the vertical through `pivot_xz`, then move them by `t_xz`."""
    r = soma.rot_y(yaw)
    pv = np.array([pivot_xz[0], 0.0, pivot_xz[1]])
    local = local.copy()
    local[:, 0] = r @ local[:, 0]
    root = (root - pv) @ r.T + pv + np.array([t_xz[0], 0.0, t_xz[1]])
    return local, root


def place_target(ctx: tuple[np.ndarray, np.ndarray], tgt: tuple[np.ndarray, np.ndarray], placement) -> tuple:
    """Target frames placed against the context's last pose: turned to its heading, then `feet` puts the mean ankle
    XZ on the context's (feet stay planted), `root` / `seam` the pelvis XZ; {dx, dz, dyaw} = `feet`, then turned by
    dyaw (deg) and moved dx (to the character's left) / dz (forward) in the context's heading frame."""
    (cl, cr), (tl, tr) = ctx, tgt
    hc, ac = _feet(cl[-1], cr[-1])
    ht, at = _feet(tl[0], tr[0])
    tl, tr = _place(tl, tr, hc - ht, tr[0, [0, 2]], (0.0, 0.0))
    if placement in ("root", "seam"):
        return _place(tl, tr, 0.0, (0, 0), cr[-1, [0, 2]] - tr[0, [0, 2]])
    _, at = _feet(tl[0], tr[0])
    tl, tr = _place(tl, tr, 0.0, (0, 0), ac - at)
    if isinstance(placement, dict):
        off = soma.rot_y(hc) @ np.array([placement["dx"], 0.0, placement["dz"]])
        tl, tr = _place(tl, tr, math.radians(placement["dyaw"]), ac, off[[0, 2]])
    return tl, tr


def body_error(a_local: np.ndarray, a_root: np.ndarray, b_local: np.ndarray, b_root: np.ndarray) -> dict:
    """One pose vs another (T-pose locals, same world frame): max local rotation over the 26 body joints (deg, with
    its joint) and pelvis distance (cm)."""
    ix = soma.rest()["index"]
    j = [ix[n] for n in BODY_JOINTS]
    ang = soma.angle_deg(a_local[j], b_local[j])
    w = int(np.argmax(ang))
    return {"rot_max_deg": round(float(ang[w]), 3), "joint": BODY_JOINTS[w],
            "root_cm": round(float(np.linalg.norm(a_root - b_root)) * 100, 3)}


# ---------------------------------------------------------------- MotionBricks


def _git_head(repo: Path) -> str | None:
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        f = repo / ".git" / ref
        if f.exists():
            return f.read_text().strip()
        for line in (repo / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref) or line.endswith(" refs/remotes/origin/" + ref.rsplit("/", 1)[-1]):
                return line.split()[0]
    except OSError:
        pass
    return None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


class MotionBricksWorker:
    def __init__(self, precision: str, cache_dir: Path):
        import warp as wp

        import motionbricks
        from motionbricks.inference.builder import TensorRTBuildOptions
        from motionbricks.io import utils as io_utils
        from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
        from motionbricks.model.soma.engine import MotionBricksSOMAEngine

        if getattr(motionbricks, "__file__", None) is None:  # `python -m` from the clone's parent folder
            raise RuntimeError(f"motionbricks imported as a namespace package from {list(motionbricks.__path__)}: "
                               "run motiongen link from another folder (the clone's folder shadows the package)")
        repo = Path(motionbricks.__file__).resolve().parents[1]
        self.wp = wp
        self.dev = wp.get_device("cuda:0")
        self.skel = io_utils.get_soma_human_model().skeleton
        names = [self.skel.joint_name(i) for i in range(self.skel.num_joints)]
        r = soma.rest()
        if sorted(names) != sorted(r["names"]):
            raise RuntimeError(f"MotionBricks skeleton is not SOMA-77 without Root ({len(names)} joints)")
        self.mb_from_k = np.array([r["index"][n] for n in names])  # MB joint i = Kimodo joint mb_from_k[i]
        self.k_from_mb = np.argsort(self.mb_from_k)
        self.seed_off = soma.bone_offsets("seed")
        onnx = SOMA_N1_FP16_ONNX_PATH if precision == "fp16" else SOMA_N1_FP32_ONNX_PATH
        t0 = time.perf_counter()
        self.eng = MotionBricksSOMAEngine(onnx, build_options=TensorRTBuildOptions(enable_fp16=precision == "fp16"),
                                          cache_dir=cache_dir, logger_severity="WARNING")
        self.load_s = time.perf_counter() - t0
        self.ctx, self.tgt = self.eng.create_input(self.skel), self.eng.create_input(self.skel)
        self.out = self.eng.create_output(self.skel)
        self.provenance = {
            "worker": "motionbricks", "model": MODEL_NAME, "precision": precision, "onnx": Path(str(onnx)).name,
            "motionbricks_version": getattr(motionbricks, "__version__", None) or _version("motionbricks"),
            "motionbricks_git": _git_head(repo), "tensorrt": _version("tensorrt-cu12") or _version("tensorrt"),
            "warp": _version("warp-lang"), "fps": MB_FPS,
        }

    def to_mb(self, local: np.ndarray, root: np.ndarray) -> np.ndarray:
        """T-pose locals + root -> MotionBricks local transforms (n, J, 7): BONES-SEED rest, p xyz (m), q xyzw."""
        q = soma.mat_to_quat(soma.from_standard_tpose(local))
        p = np.broadcast_to(self.seed_off, local.shape[:2] + (3,)).copy()
        p[:, 0] = root
        return np.concatenate([p, q], -1)[:, self.mb_from_k].astype(np.float32)

    def from_mb(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = poses[:, self.k_from_mb].astype(np.float64)
        return soma.to_standard_tpose(soma.quat_to_mat(k[..., 3:])), k[:, 0, :3]

    def generate(self, ctx_mb: np.ndarray, tgt_mb: np.ndarray, tokens: tuple[int, ...], correct: bool) -> tuple:
        from motionbricks.model.config import MotionBricksInferenceConfig

        wp = self.wp
        self.ctx.update_from_poses(wp.array(ctx_mb, dtype=wp.transform, device=self.dev), stream=self.eng.stream)
        self.tgt.update_from_poses(wp.array(tgt_mb, dtype=wp.transform, device=self.dev), stream=self.eng.stream)
        cfg = MotionBricksInferenceConfig(allowed_token_counts=tokens, correct_generated_diff=correct)
        t = time.perf_counter()
        self.eng.generate_animation(self.ctx, self.tgt, self.out, config=cfg)
        self.eng.synchronize()
        ms = (time.perf_counter() - t) * 1e3
        n = self.out.num_frames
        return self.out.poses.numpy()[:n].copy(), self.out.foot_contacts.numpy()[:n].copy(), ms

    def close(self) -> None:
        self.eng.close()


# ---------------------------------------------------------------- jobs


def _ref_prov(ref: PoseRef, path: Path, frame: int, end: str) -> dict:
    if ref.hub:
        return {f"{end}_hub": ref.hub}
    return {f"{end}_clip": {"path": str(path), "frame": frame}}


def run_job(job: LinkJob, worker: MotionBricksWorker, sources: Sources, out: Path) -> dict:
    t0 = time.perf_counter()
    sl, sr, spath = sources.clip(job.src)
    dl, dr, dpath = sources.clip(job.dst)
    k = Sources.frame_of(job.src, len(sl))
    h = Sources.frame_of(job.dst, len(dl))
    ci = [max(0, k - POSES_PER_TOKEN + 1 + i) for i in range(POSES_PER_TOKEN)]
    ti = [min(len(dl) - 1, h + i) for i in range(job.target_poses)]
    ctx = (sl[ci], sr[ci])
    tgt = place_target(ctx, (dl[ti], dr[ti]), job.placement)
    ctx_mb, tgt_mb = worker.to_mb(*ctx), worker.to_mb(*tgt)
    raw, _, raw_ms = worker.generate(ctx_mb, tgt_mb, job.tokens, correct=False)
    poses, contacts, ms = worker.generate(ctx_mb, tgt_mb, job.tokens, correct=True)
    again, _, _ = worker.generate(ctx_mb, tgt_mb, job.tokens, correct=True)
    local, root = worker.from_mb(poses)
    rl, rr = worker.from_mb(raw)
    bvh, prov_path = job.outputs(out)
    bvh.parent.mkdir(parents=True, exist_ok=True)
    soma.write_bvh(bvh, local, root, MB_FPS)
    n = len(local)
    stats = {
        "frames": n, "tokens": (n + TRIM_FRAMES) // POSES_PER_TOKEN, "seconds": round(n / MB_FPS, 3),
        "start_error": body_error(local[0], root[0], ctx[0][-1], ctx[1][-1]),
        "end_error": body_error(local[-1], root[-1], tgt[0][0], tgt[1][0]),
        "correction": {"start": body_error(rl[0], rr[0], ctx[0][-1], ctx[1][-1]),
                       "end": body_error(rl[-1], rr[-1], tgt[0][0], tgt[1][0]),
                       "raw_frames": len(rl)},
        "deterministic": bool(again.shape == poses.shape and np.abs(again - poses).max() < 1e-5),
        "generate_ms": round(ms, 2), "raw_generate_ms": round(raw_ms, 2),
    }
    prov = {
        "tool": "motiongen", "tool_version": TOOL_VERSION, **worker.provenance,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "id": job.stem(), "link": job.id, "kind": job.kind, "take": job.take,
        **_ref_prov(job.src, spath, k, "from"), **_ref_prov(job.dst, dpath, h, "to"),
        "context_frames": ci, "target_frames": ti, "target_poses": job.target_poses,
        "allowed_tokens": list(job.tokens), "placement": job.placement, "endpoint_correction": True,
        **stats,
        "recipe": job.source,
        "bvh": {"skeleton": "somaskel77", "units": "cm", "up": "+Y", "rest_pose": "standard T-pose",
                "hips_translation": "absolute (cm); the Hips OFFSET is a dummy (0,100,0)"},
        "contacts": bvh.with_suffix(".contacts.json").name,
        "reproduce": "same sources, frames, placement, allowed_tokens, model precision (deterministic: see field)",
    }
    bvh.with_suffix(".contacts.json").write_text(json.dumps({
        "columns": ["heel_l", "toe_l", "heel_r", "toe_r"], "fps": MB_FPS,
        "contacts": np.round(contacts, 3).tolist()}), encoding="utf-8")
    tmp = prov_path.with_name(prov_path.name + ".tmp")
    tmp.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    os.replace(tmp, prov_path)
    stats["wall_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return stats


def _row(job: LinkJob, s: dict) -> str:
    c = s["correction"]
    e = lambda d: f"{d['rot_max_deg']:.2f}/{d['root_cm']:.2f}"
    return (f"| {job.stem()} | {job.src.label()} -> {job.dst.label()} | {s['frames']} ({s['tokens']}t) | "
            f"{e(s['start_error'])} | {e(s['end_error'])} | {c['start']['rot_max_deg']:.1f} {c['start']['joint']} / "
            f"{c['start']['root_cm']:.1f} | {c['end']['rot_max_deg']:.1f} {c['end']['joint']} / {c['end']['root_cm']:.1f} | "
            f"{s['generate_ms']:.1f} | {'yes' if s['deterministic'] else 'NO'} |")


def cmd_link(args) -> int:
    out = Path(args.out)
    hub_root = Path(args.hub_root) if args.hub_root else out
    jobs = plan_links(load_links([Path(p) for p in args.recipes]), hub_root, args.only)
    todo = [j for j in jobs if args.force or not j.done(out)]
    if args.plan:
        for j in jobs:
            print(link_plan_line(j, "done" if j.done(out) else "todo"))
        return 0
    print(f"[motiongen link] {len(jobs)} takes planned, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    worker = MotionBricksWorker(args.precision, Path(args.trt_cache))
    print(f"[motiongen link] engine {MODEL_NAME} {args.precision} loaded in {worker.load_s:.2f} s")
    sources = Sources(hub_root)
    rows = ["| take | from -> to | frames | start err deg/cm | end err deg/cm | correction start deg joint / cm | "
            "correction end deg joint / cm | ms | deterministic |", "|---|---|---|---|---|---|---|---|---|"]
    try:
        for i, job in enumerate(todo):
            s = run_job(job, worker, sources, out)
            rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "job": job.stem(), "kind": job.kind,
                   "from": job.src.label(), "to": job.dst.label(), **s}
            with (out / "jobs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            rows.append(_row(job, s))
            print(f"[{i + 1}/{len(todo)}] {job.stem()}: {s['frames']} frames, {s['generate_ms']:.1f} ms", flush=True)
    finally:
        worker.close()
    print("\n".join(rows))
    print(f"[motiongen link] {len(todo)} takes in {time.perf_counter() - t_start:.1f} s wall "
          f"(engine load {worker.load_s:.2f} s); first generation includes the CUDA graph capture")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("link", help="MotionBricks link pieces (approach, loop seams, exit) between given poses (MG8)")
    p.add_argument("recipes", nargs="+", help="link recipe TOML files ([[link]] / [[loop_seam]] tables)")
    p.add_argument("--out", required=True, help="output root; takes go to <out>/<id>/<id>.<nn>.bvh")
    p.add_argument("--hub-root", help="root that relative hub sources and clip paths resolve against (default --out)")
    p.add_argument("--only", action="append", help="link id glob (repeatable)")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="TensorRT plan (fp16 = the cached one)")
    p.add_argument("--trt-cache", default=str(Path(os.environ.get("MOTIONGEN_MOTIONBRICKS") or DEFAULT_MB_ROOT) / ".trt-cache"),
                   help="TensorRT plan cache (a missing plan builds once: ~7 min, ~7 GB host RAM)")
    p.add_argument("--plan", action="store_true", help="dry run: list the takes, no GPU")
    p.add_argument("--force", action="store_true", help="regenerate takes whose outputs exist")
    p.set_defaults(fn=cmd_link)
