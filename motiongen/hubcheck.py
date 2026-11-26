# SPDX-License-Identifier: GPL-3.0-or-later
"""`python -m motiongen hubcheck <dir>` (AN44b, numpy only, no Kimodo): how close generated takes start and end on
their hub poses (`recipes/hubs.toml`), plus foot slide over the take. Printed measurements, never pass/fail.

A take is checked when its `.prov.json` names `from_hub` / `to_hub` (AN44a) or `from_clip` / `to_clip` ({path, frame}:
MG8 links and loop seams, which also add a correction-size table); others are skipped. Per checked end
(first frame vs `from_hub`, last frame vs `to_hub`; the hub pose = frame `frame` of its SOMA-77 `source` BVH):
- rotation: per-joint local rotation error (deg, geodesic) over the 26 body joints (the model's SOMA-30 set minus its
  four hand End joints: no fingers), max and mean. The pelvis (Hips) rotation is compared with each pose's heading
  removed. Local rotations are compared as written in the BVHs, so take and hub must share a rest pose (same joint
  offsets; the angle is then the same in either rest convention). A rest mismatch leaves rotation unmeasured.
- position: hand L/R, foot L/R and pelvis error (cm) in the character frame: pelvis XZ at the origin, heading (thigh
  roots, as `grade.heading_deg`) turned to +Z, height kept; `worst_cm` = the largest. `pelvis_dy_cm` = signed height
  error (take - hub).
Foot slide: per foot and frame, the horizontal speed of its slowest point (ankle, toe) among those in contact on both
ends of the step (contact = height under the point's resting height in the take, its 2nd percentile, + 3 cm: grade's
contact band); mean and p95 over all such foot-frames (cm/s). Rotation channels: `clipstats._rot` (BVH order).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .clipstats import Bvh, _rot, collect, forward_kinematics, parse_bvh
from .grade import UP, group_of_stem, write_json_atomic
from .hubs import Hub, HubError, default_hubs

# SOMA-30 (Kimodo's model skeleton) without its four hand End joints = the 26 body joints (no fingers).
BODY_JOINTS = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)
EFFECTORS = {"hand_l": "LeftHand", "hand_r": "RightHand", "foot_l": "LeftFoot", "foot_r": "RightFoot", "pelvis": "Hips"}
FOOT_POINTS = {"l": ("LeftFoot", "LeftToeBase"), "r": ("RightFoot", "RightToeBase")}
CONTACT_BAND_M = 0.03  # grade.STEP_BAND_M: foot point within 3 cm of its resting height = in contact
REST_TOL_CM = 0.1  # joint offsets (cm, BVH units) equal within this = same rest pose


class Take:
    """One parsed SOMA-77 BVH: local rotations (T, J, 3, 3), world positions (T, J, 3) in m."""

    def __init__(self, b: Bvh):
        self.b = b
        self.names = b.names
        self.index = {n: j for j, n in enumerate(b.names)}
        self.fps = 1.0 / b.frame_time
        self.pos = forward_kinematics(b, 0.01, "channels")
        T, col = b.motion.shape[0], 0
        self.local = np.zeros((T, len(b.names), 3, 3))
        for j, chans in enumerate(b.channels):
            r = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
            for k, ch in enumerate(chans):
                if ch.endswith("rotation"):
                    r = r @ _rot(ch[0], b.motion[:, col + k])
            self.local[:, j] = r
            col += len(chans)

    @property
    def frames(self) -> int:
        return self.pos.shape[0]

    def heading(self, k: int) -> float:
        """Facing angle (rad) at frame k: + = toward +X from +Z (same convention as `grade.heading_deg`)."""
        left = self.pos[k, self.index["LeftLeg"]] - self.pos[k, self.index["RightLeg"]]
        fwd = np.cross(left, UP)
        return math.atan2(fwd[0], fwd[2])

    def char_frame(self, k: int) -> np.ndarray:
        """Joint positions (J, 3) at frame k in the character frame: pelvis XZ at 0, heading to +Z, height kept."""
        p = self.pos[k] - self.pos[k, self.index["Hips"]] * np.array([1.0, 0.0, 1.0])
        return p @ _rot("Y", np.array(-math.degrees(self.heading(k)))).T

    def body_local(self, k: int) -> dict[str, np.ndarray]:
        """Local rotations of the body joints present at frame k; the pelvis with its heading removed."""
        out = {n: self.local[k, self.index[n]] for n in BODY_JOINTS if n in self.index}
        if "Hips" in out:
            h = self.index["Hips"]
            p = self.b.parents[h]
            parent = self.local[k, p] if p >= 0 else np.eye(3)  # Kimodo's wrapper 'Root' is identity
            out["Hips"] = _rot("Y", np.array(-math.degrees(self.heading(k)))) @ parent @ self.local[k, h]
        return out

    def offsets(self) -> dict[str, np.ndarray]:
        """Rest offsets of the non-pelvis body joints (cm); the pelvis' OFFSET is a dummy in Kimodo BVHs."""
        return {n: self.b.offsets[self.index[n]] for n in BODY_JOINTS if n in self.index and n != "Hips"}


def rot_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices (deg): ||A - B||_F = 2 sqrt(2) sin(theta / 2), exact near 0."""
    s = np.linalg.norm(a - b) / (2.0 * math.sqrt(2.0))
    return math.degrees(2.0 * math.asin(min(1.0, s)))


def same_rest(a: Take, b: Take) -> bool:
    oa, ob = a.offsets(), b.offsets()
    return set(oa) == set(ob) and all(np.abs(oa[n] - ob[n]).max() <= REST_TOL_CM for n in oa)


def pose_error(take: Take, k: int, hub: Take, h: int) -> dict:
    """Take frame k vs hub frame h: rotation (deg) and effector (cm) errors, see the module doc."""
    out: dict = {}
    if same_rest(take, hub):
        ra, rb = take.body_local(k), hub.body_local(h)
        errs = {n: rot_angle_deg(ra[n], rb[n]) for n in ra if n in rb}
        worst = max(errs, key=errs.get)
        out.update(rot_max_deg=round(errs[worst], 2), rot_mean_deg=round(float(np.mean(list(errs.values()))), 2),
                   rot_worst_joint=worst, joints=len(errs))
    else:
        out.update(rot_max_deg=None, rot_mean_deg=None, rest_mismatch=True)
    pa, pb = take.char_frame(k), hub.char_frame(h)
    eff = {key: round(float(np.linalg.norm(pa[take.index[n]] - pb[hub.index[n]])) * 100, 1) for key, n in EFFECTORS.items()}
    out["effector_cm"] = eff
    out["worst_cm"] = max(eff.values())
    out["pelvis_dy_cm"] = round(float(pa[take.index["Hips"], 1] - pb[hub.index["Hips"], 1]) * 100, 1)
    return out


def foot_slide(t: Take) -> dict:
    """Foot slide over the take (cm/s): mean and p95 of in-contact foot speeds, plus the contact share."""
    speeds: list[np.ndarray] = []
    contact = 0
    for pts in FOOT_POINTS.values():
        cm, st = [], []
        for n in pts:
            p = t.pos[:, t.index[n]]
            y = p[:, 1]
            cm.append(y < max(0.0, float(np.percentile(y, 2))) + CONTACT_BAND_M)
            st.append(np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1))
        cm_a = np.array(cm)
        both = cm_a[:, 1:] & cm_a[:, :-1]
        step = np.where(both, np.array(st), np.inf).min(axis=0)
        speeds.append(step[np.isfinite(step)] * t.fps * 100)
        contact += int(both.any(axis=0).sum())
    v = np.concatenate(speeds) if speeds else np.zeros(0)
    if v.size == 0:
        return {"slide_mean_cm_s": None, "slide_p95_cm_s": None, "contact_share": 0.0}
    return {"slide_mean_cm_s": round(float(v.mean()), 1), "slide_p95_cm_s": round(float(np.percentile(v, 95)), 1),
            "contact_share": round(contact / (2 * max(1, t.frames - 1)), 2)}


class HubFrames:
    """Hub id -> (parsed source take, frame), each source BVH parsed once."""

    def __init__(self, hubs: dict[str, Hub], root: Path):
        self.hubs, self.root, self.cache = hubs, root, {}

    def get(self, hid: str) -> tuple[Take, int]:
        if hid not in self.hubs:
            raise HubError(f"no hub '{hid}' in the hubs table")
        hub = self.hubs[hid]
        path = hub.source_path(self.root)
        if path not in self.cache:
            if not path.exists():
                raise HubError(f"hub '{hid}': source {path} not found (relative sources resolve against --hub-root)")
            self.cache[path] = Take(parse_bvh(path))
        take = self.cache[path]
        if hub.frame >= take.frames:
            raise HubError(f"hub '{hid}': frame {hub.frame} outside {path} ({take.frames} frames)")
        return take, hub.frame

    def clip(self, ref: dict) -> tuple[Take, int]:
        """A link end on a clip frame (`from_clip` / `to_clip` = {path, frame}, MG8 links and loop seams)."""
        path = Path(ref["path"])
        if path not in self.cache:
            if not path.exists():
                raise HubError(f"clip {path} not found")
            self.cache[path] = Take(parse_bvh(path))
        take, k = self.cache[path], int(ref["frame"])
        if not 0 <= k < take.frames:
            raise HubError(f"clip {path}: frame {k} outside ({take.frames} frames)")
        return take, k


def _end_label(prov: dict, end: str) -> str | None:
    if prov.get(f"{end}_hub"):
        return prov[f"{end}_hub"]
    c = prov.get(f"{end}_clip")
    return f"{Path(c['path']).name}#{c['frame']}" if isinstance(c, dict) else None


def check_take(path: Path, prov: dict, hubs: HubFrames) -> dict:
    t = Take(parse_bvh(path))
    out: dict = {"from_hub": _end_label(prov, "from"), "to_hub": _end_label(prov, "to"), "frames": t.frames}
    for end, key, k in (("start", "from", 0), ("end", "to", t.frames - 1)):
        if prov.get(f"{key}_hub") or prov.get(f"{key}_clip"):
            try:
                hub, h = hubs.get(prov[f"{key}_hub"]) if prov.get(f"{key}_hub") else hubs.clip(prov[f"{key}_clip"])
                out[end] = pose_error(t, k, hub, h)
            except HubError as e:
                out[end] = {"error": str(e)}
    out.update(foot_slide(t))
    if isinstance(prov.get("correction"), dict):  # MG8 link takes: endpoint correction size + generation time
        c = prov["correction"]
        out["link"] = {"tokens": prov.get("tokens"), "ms": prov.get("generate_ms"),
                       "corr_start_deg": c["start"]["rot_max_deg"], "corr_start_joint": c["start"]["joint"],
                       "corr_start_cm": c["start"]["root_cm"], "corr_end_deg": c["end"]["rot_max_deg"],
                       "corr_end_joint": c["end"]["joint"], "corr_end_cm": c["end"]["root_cm"]}
    return out


def hubcheck_dir(root: Path, hubs: dict[str, Hub], hub_root: Path) -> dict:
    """{stem: result} for every hub take under `root`, `_groups` (per prompt group) and `_skipped` (no hub keys)."""
    frames = HubFrames(hubs, hub_root)
    out: dict = {}
    skipped: list[str] = []
    for f in collect([str(root)]):
        pp = f.with_suffix(".prov.json")
        prov = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
        if not any(prov.get(k) for k in ("from_hub", "to_hub", "from_clip", "to_clip")):
            skipped.append(f.stem)
            continue
        out[f.stem] = check_take(f, prov, frames)
    groups: dict[str, list[dict]] = {}
    for stem, r in out.items():
        groups.setdefault(group_of_stem(stem), []).append(r)
    out["_groups"] = {g: group_summary(rs) for g, rs in groups.items()}
    out["_skipped"] = sorted(skipped)
    return out


def _vals(rs: list[dict], end: str, key: str) -> list[float]:
    return [r[end][key] for r in rs if isinstance(r.get(end), dict) and r[end].get(key) is not None]


def group_summary(rs: list[dict]) -> dict:
    """Per prompt group: take count, then mean and max of each end's max rotation and worst effector, slide p95."""
    def mm(v: list[float]) -> list[float] | None:
        return [round(float(np.mean(v)), 2), round(float(max(v)), 2)] if v else None
    return {"takes": len(rs),
            "start_rot_max_deg": mm(_vals(rs, "start", "rot_max_deg")), "start_worst_cm": mm(_vals(rs, "start", "worst_cm")),
            "end_rot_max_deg": mm(_vals(rs, "end", "rot_max_deg")), "end_worst_cm": mm(_vals(rs, "end", "worst_cm")),
            "slide_p95_cm_s": mm([r["slide_p95_cm_s"] for r in rs if r.get("slide_p95_cm_s") is not None])}


def _f(v, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def _end_cells(r: dict, end: str) -> list[str]:
    e = r.get(end)
    if not isinstance(e, dict):
        return ["-", "-"]
    if "error" in e:
        return ["error", "-"]
    rot = "rest?" if e.get("rest_mismatch") else f"{e['rot_max_deg']:.1f}/{e['rot_mean_deg']:.1f}"
    return [rot, f"{e['worst_cm']:.1f}"]


def table(res: dict) -> list[str]:
    rows = ["| take | from -> to | start max/mean deg | start worst cm | end max/mean deg | end worst cm | slide mean/p95 cm/s |",
            "|---|---|---|---|---|---|---|"]
    for stem, r in sorted((k, v) for k, v in res.items() if not k.startswith("_")):
        hubs = f"{r['from_hub'] or 'none'} -> {r['to_hub'] or 'none'}"
        slide = f"{_f(r['slide_mean_cm_s'], '.1f')}/{_f(r['slide_p95_cm_s'], '.1f')}"
        rows.append("| " + " | ".join([stem, hubs, *_end_cells(r, "start"), *_end_cells(r, "end"), slide]) + " |")
    rows += ["", "| group | takes | start max deg mean/max | start worst cm mean/max | end max deg mean/max | end worst cm mean/max | slide p95 mean/max |",
             "|---|---|---|---|---|---|---|"]
    for g, s in sorted(res["_groups"].items()):
        cell = lambda v: "-" if v is None else f"{v[0]:.1f}/{v[1]:.1f}"
        rows.append(f"| {g} | {s['takes']} | " + " | ".join(cell(s[k]) for k in (
            "start_rot_max_deg", "start_worst_cm", "end_rot_max_deg", "end_worst_cm", "slide_p95_cm_s")) + " |")
    links = sorted((k, v["link"]) for k, v in res.items() if not k.startswith("_") and "link" in v)
    if links:
        rows += ["", "| link take | tokens | correction start deg joint / pelvis cm | correction end deg joint / pelvis cm | ms/segment |",
                 "|---|---|---|---|---|"]
        for stem, s in links:
            rows.append(f"| {stem} | {s['tokens']} | {s['corr_start_deg']:.1f} {s['corr_start_joint']} / {s['corr_start_cm']:.1f} | "
                        f"{s['corr_end_deg']:.1f} {s['corr_end_joint']} / {s['corr_end_cm']:.1f} | {_f(s['ms'], '.1f')} |")
    for stem, r in sorted((k, v) for k, v in res.items() if not k.startswith("_")):
        for end in ("start", "end"):
            if isinstance(r.get(end), dict) and "error" in r[end]:
                rows.append(f"{stem} {end}: {r[end]['error']}")
    return rows


def cmd_hubcheck(args) -> int:
    root = Path(args.dir)
    if not collect([str(root)]):
        print(f"motiongen hubcheck: no .bvh files under {root}")
        return 1
    hub_root = Path(args.hub_root) if args.hub_root else root.resolve().parent
    res = hubcheck_dir(root, default_hubs(), hub_root)
    out = Path(args.out) if args.out else root / "hubcheck.json"
    write_json_atomic(out, res)
    print("\n".join(table(res)))
    print(f"{len(res) - 2} hub takes, {len(res['_skipped'])} skipped (no from/to hub or clip); wrote {out}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("hubcheck", help="start/end error vs the hub poses + foot slide for hub takes -> hubcheck.json")
    p.add_argument("dir", help="folder of generated takes (recursive) whose .prov.json name from_hub / to_hub")
    p.add_argument("--hub-root", help="root that relative hub sources of recipes/hubs.toml resolve against (default <dir>'s parent)")
    p.add_argument("--out", help="output path (default <dir>/hubcheck.json)")
    p.set_defaults(fn=cmd_hubcheck)
