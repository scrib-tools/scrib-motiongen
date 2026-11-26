# SPDX-License-Identifier: GPL-3.0-or-later
"""Link recipes (MG8): `[[link]]` and `[[loop_seam]]` tables -> MotionBricks link jobs. Pure Python (no GPU).

A link is one MotionBricks in-betweening from a context pose (the last 4 frames up to `from`) to a target pose (the
first `target_poses` frames from `to`); a loop seam links a clip's end back to its own start; a hub loop (AN26a) is a
loop take made exact on a hub: link hub -> clip start, the clip, link clip end -> the same hub, joined in one take. Format: README "link".
Pose sources are hubs (`recipes/hubs.toml`) or kept SOMA-77 takes; poses from MotionBricks' bundled BONES samples are
refused (NVIDIA Sample Data Evaluation License: evaluation only, no distributed derivatives).
"""

from __future__ import annotations

import fnmatch
import importlib.util
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hubs import Hub, HubError, default_hubs
from .recipe import ID_RE, RecipeError

MB_FPS = 30  # MotionBricks SOMA model space (30 Hz)
POSES_PER_TOKEN = 4  # model contract: a token holds 4 poses
MIN_TOKENS, MAX_TOKENS = 6, 16  # model contract (`MODEL_MIN/MAX_TOKEN_COUNT`)
TRIM_FRAMES = 6  # engine drops 3 boundary frames at each end: valid frames = 4 * tokens - 6
DEFAULT_TOKENS = (8, 14)  # MotionBricksInferenceConfig default `allowed_token_counts`
SEAM_TOKENS = 6  # loop seams: shortest link (18 frames), the AN13 spike setting
LINK_KEYS = {"id", "from", "to", "tokens", "seconds", "placement", "target_poses", "variants", "notes"}
SEAM_KEYS = {"id", "clip", "tokens", "seconds", "target_poses", "variants", "notes"}
HUB_LOOP_KEYS = {"id", "hub", "clip", "tokens", "seconds", "target_poses", "variants", "notes"}
HUB_LOOP_TOKENS = (6, 10)  # hub loops: two short links (hub -> clip start, clip end -> hub), the model picks
VARIANT_KEYS = {"tokens", "seconds", "placement", "target_poses"}
PLACEMENTS = ("feet", "root")
CLIP_RE = re.compile(r"^clip:(?P<path>.+?)#(?P<at>start|end|\d+)$")
BONES_NAME_RE = re.compile(r"__A\d+(_M)?\.bvh$", re.IGNORECASE)  # BONES-SEED take naming (`..._001__A549.bvh`)
DEFAULT_MB_ROOT = Path(r"C:\mbtest\motionbricks")


def link_frames(tokens: int) -> int:
    return POSES_PER_TOKEN * tokens - TRIM_FRAMES


def motionbricks_roots() -> list[Path]:
    """MotionBricks checkouts whose sample folders are refused: $MOTIONGEN_MOTIONBRICKS, the importable package's
    repo, and the default clone."""
    roots = [Path(os.environ["MOTIONGEN_MOTIONBRICKS"])] if os.environ.get("MOTIONGEN_MOTIONBRICKS") else []
    try:
        spec = importlib.util.find_spec("motionbricks")
        if spec and spec.origin:
            roots.append(Path(spec.origin).resolve().parents[1])
    except (ImportError, ValueError):
        pass
    roots.append(DEFAULT_MB_ROOT)
    return roots


def refuse_sample(path: Path, where: str) -> None:
    """Raise if `path` is a MotionBricks sample motion (evaluation-only BONES data, AN13 §5)."""
    p = Path(os.path.abspath(path))
    for root in motionbricks_roots():
        for sub in ("assets/motions", "motionbricks/assets/motions"):
            base = Path(os.path.abspath(root / sub))
            if p == base or base in p.parents:
                raise RecipeError(f"{where}: {path} is a MotionBricks sample motion ({base}): NVIDIA Sample Data "
                                  "Evaluation License, evaluation only; take poses from our own takes (hubs.toml, Kimodo output)")
    if BONES_NAME_RE.search(p.name):
        raise RecipeError(f"{where}: {path} is named like a BONES-SEED take (evaluation-only sample data); "
                          "take poses from our own takes (hubs.toml, Kimodo output)")


@dataclass(frozen=True)
class PoseRef:
    """One end of a link: a hub (frame from hubs.toml) or a clip frame (`start` = 0, `end` = last, or an index)."""
    kind: str  # "hub" | "clip"
    path: str  # BVH as written (hub source or clip path)
    at: str  # "start" | "end" | frame index as text
    hub: str | None = None

    def resolve(self, root: Path | None) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else (Path(root) if root is not None else Path.cwd()) / p

    def label(self) -> str:
        return f"hub:{self.hub}" if self.hub else f"clip:{Path(self.path).name}#{self.at}"


def _err(src: str, where: str, msg: str) -> RecipeError:
    return RecipeError(f"{src}: {where}: {msg}")


def _pose_ref(src: str, where: str, v: Any, end: str, hubs) -> PoseRef:
    if isinstance(v, str) and v.startswith("hub:"):
        hid = v[4:].strip()
        try:
            table = default_hubs() if hubs is None else hubs
        except HubError as e:
            raise _err(src, where, f"cannot load the hub table: {e}") from e
        if hid not in table:
            raise _err(src, where, f"no hub '{hid}' (hubs: {sorted(table)})")
        h: Hub = table[hid]
        return PoseRef("hub", h.source, str(h.frame), hid)
    m = CLIP_RE.match(v) if isinstance(v, str) else None
    if not m or not m["path"].lower().endswith(".bvh"):
        raise _err(src, where, f"{v!r}: expected 'hub:<id>' or 'clip:<path.bvh>#{end}' (or '#<frame>')")
    return PoseRef("clip", m["path"], m["at"])


def _tokens(src: str, where: str, t: dict) -> tuple[int, ...] | None:
    if "tokens" in t and "seconds" in t:
        raise _err(src, where, "give tokens or seconds, not both")
    if "seconds" in t:
        s = t["seconds"]
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            raise _err(src, where, "seconds must be a number")
        n = round((s * MB_FPS + TRIM_FRAMES) / POSES_PER_TOKEN)
        if not MIN_TOKENS <= n <= MAX_TOKENS:
            raise _err(src, where, f"seconds = {s}: outside {link_frames(MIN_TOKENS) / MB_FPS:.2f}.."
                                   f"{link_frames(MAX_TOKENS) / MB_FPS:.2f} s (one MotionBricks segment)")
        return (n,)
    if "tokens" not in t:
        return None
    v = t["tokens"]
    ok = lambda x: isinstance(x, int) and not isinstance(x, bool) and MIN_TOKENS <= x <= MAX_TOKENS
    if ok(v):
        return (v,)
    if isinstance(v, list) and len(v) == 2 and all(ok(x) for x in v) and v[0] <= v[1]:
        return tuple(range(v[0], v[1] + 1))
    raise _err(src, where, f"tokens = {v!r}: an integer {MIN_TOKENS}..{MAX_TOKENS} (exact) or [lo, hi] (model picks)")


def _placement(src: str, where: str, v: Any) -> str | dict:
    if v in PLACEMENTS:
        return v
    if isinstance(v, dict) and v and not set(v) - {"dx", "dz", "dyaw"} and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v.values()):
        return {k: float(v.get(k, 0.0)) for k in ("dx", "dz", "dyaw")}
    raise _err(src, where, f"placement = {v!r}: 'feet', 'root' or {{ dx = m, dz = m, dyaw = deg }}")


def _target_poses(src: str, where: str, v: Any) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= POSES_PER_TOKEN:
        raise _err(src, where, f"target_poses = {v!r}: 1..{POSES_PER_TOKEN} (4 keeps the target's velocity)")
    return v


@dataclass
class LinkJob:
    id: str
    take: int
    kind: str  # "link" | "loop_seam" | "hub_loop"
    src: PoseRef  # context: the 4 frames up to it
    dst: PoseRef  # target: the frames from it
    tokens: tuple[int, ...]
    placement: str | dict  # "feet" | "root" | "seam" | {dx, dz, dyaw}
    target_poses: int
    source: str  # recipe file

    def stem(self) -> str:
        return f"{self.id}.{self.take:02d}"

    def outputs(self, root: Path) -> tuple[Path, Path]:
        d = Path(root) / self.id
        return d / f"{self.stem()}.bvh", d / f"{self.stem()}.prov.json"

    def done(self, root: Path) -> bool:
        return all(p.exists() for p in self.outputs(root))

    def frames_label(self) -> str:
        a, b = link_frames(self.tokens[0]), link_frames(self.tokens[-1])
        return f"{a}f" if a == b else f"{a}-{b}f"


@dataclass
class LinkRecipe:
    id: str
    kind: str
    src: PoseRef
    dst: PoseRef
    tokens: tuple[int, ...]
    placement: str | dict
    target_poses: int
    variants: list[dict] = field(default_factory=list)
    source: str = ""
    more_dst: list[PoseRef] = field(default_factory=list)  # hub_loop: further clips (one take each per variant)


def parse_links_text(text: str, src: str, hubs=None) -> list[LinkRecipe]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RecipeError(f"{src}: TOML error: {e}") from e
    if set(data) - {"link", "loop_seam", "hub_loop"}:
        raise RecipeError(f"{src}: unknown top-level keys {sorted(set(data) - {'link', 'loop_seam', 'hub_loop'})} "
                          "(a link recipe has [[link]], [[loop_seam]] and [[hub_loop]] tables)")
    out: list[LinkRecipe] = []
    for kind, keys in (("link", LINK_KEYS), ("loop_seam", SEAM_KEYS), ("hub_loop", HUB_LOOP_KEYS)):
        for i, t in enumerate(data.get(kind, [])):
            where = f"{kind}[{i}]"
            if set(t) - keys:
                raise _err(src, where, f"unknown keys {sorted(set(t) - keys)} (allowed: {sorted(keys)})")
            lid = t.get("id")
            if not isinstance(lid, str) or not ID_RE.match(lid):
                raise _err(src, where, f"id {lid!r} must be dotted lowercase")
            where = f"{kind} '{lid}'"
            if kind == "link":
                for req in ("from", "to"):
                    if req not in t:
                        raise _err(src, where, f"missing '{req}'")
                a = _pose_ref(src, f"{where}.from", t["from"], "end", hubs)
                b = _pose_ref(src, f"{where}.to", t["to"], "start", hubs)
                placement = _placement(src, f"{where}.placement", t.get("placement", "feet"))
                default_tokens = DEFAULT_TOKENS
            else:
                clip = t.get("clip")
                clips = clip if kind == "hub_loop" and isinstance(clip, list) and clip else [clip]
                if not all(isinstance(c, str) and c.lower().endswith(".bvh") for c in clips):
                    raise _err(src, f"{where}.clip", f"{clip!r}: expected a .bvh path (a loop take)"
                               + (" or a list of them (one take each, in order)" if kind == "hub_loop" else ""))
                clip = clips[0]
                if kind == "loop_seam":
                    a, b = PoseRef("clip", clip, "end"), PoseRef("clip", clip, "start")
                    placement, default_tokens = "seam", (SEAM_TOKENS,)
                else:  # hub_loop: src = the hub, dst = the clip (its start; its end links back to the hub)
                    hub = t.get("hub")
                    if not isinstance(hub, str) or hub.startswith(("hub:", "clip:")):
                        raise _err(src, f"{where}.hub", f"{hub!r}: expected a hub id (recipes/hubs.toml), e.g. 'sit.chair'")
                    a, b = _pose_ref(src, f"{where}.hub", f"hub:{hub}", "end", hubs), PoseRef("clip", clip, "start")
                    placement, default_tokens = "hub_loop", HUB_LOOP_TOKENS
            tokens = _tokens(src, where, t)
            tokens = tuple(range(default_tokens[0], default_tokens[-1] + 1)) if tokens is None else tokens
            tp = _target_poses(src, f"{where}.target_poses", t.get("target_poses", POSES_PER_TOKEN))
            variants = t.get("variants", [{}])
            if not isinstance(variants, list) or not variants or not all(isinstance(v, dict) for v in variants):
                raise _err(src, f"{where}.variants", "must be a non-empty list of tables (overrides per take)")
            for k, v in enumerate(variants):
                w = f"{where}.variants[{k}]"
                if set(v) - VARIANT_KEYS:
                    raise _err(src, w, f"unknown keys {sorted(set(v) - VARIANT_KEYS)} (allowed: {sorted(VARIANT_KEYS)})")
                if "placement" in v and kind != "link":
                    raise _err(src, w, f"a {kind}'s placement is fixed (the clip at the hub's pelvis, the hub back at the start)"
                               if kind == "hub_loop" else "a loop seam's placement is fixed (target = the clip start at the end root)")
                _tokens(src, w, v)
                if "placement" in v:
                    _placement(src, w, v["placement"])
                if "target_poses" in v:
                    _target_poses(src, w, v["target_poses"])
            extra = [PoseRef("clip", c, "start") for c in clips[1:]] if kind == "hub_loop" else []
            out.append(LinkRecipe(lid, kind, a, b, tokens, placement, tp, variants, src, extra))
    if not out:
        raise RecipeError(f"{src}: no [[link]], [[loop_seam]] or [[hub_loop]] tables")
    ids = [r.id for r in out]
    dups = sorted({x for x in ids if ids.count(x) > 1})
    if dups:
        raise RecipeError(f"{src}: duplicate link ids {dups}")
    return out


def load_links(paths: list[Path], hubs=None) -> list[LinkRecipe]:
    out: list[LinkRecipe] = []
    seen: dict[str, str] = {}
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except OSError as e:
            raise RecipeError(f"{p}: {e}") from e
        for r in parse_links_text(text, str(p), hubs):
            if r.id in seen:
                raise RecipeError(f"{p}: link '{r.id}' already defined in {seen[r.id]}")
            seen[r.id] = str(p)
            out.append(r)
    return out


def plan_links(recipes: list[LinkRecipe], hub_root: Path | None, only: list[str] | None = None) -> list[LinkJob]:
    """One job per take (variant). Every pose source is resolved against `hub_root` and checked for sample data."""
    jobs = []
    for r in recipes:
        if only and not any(fnmatch.fnmatchcase(r.id, pat) for pat in only):
            continue
        for ref, end in ((r.src, "from"), *((d, "to") for d in [r.dst, *r.more_dst])):
            refuse_sample(ref.resolve(hub_root), f"{r.source}: {r.kind} '{r.id}'.{end}")
        n = 0
        for dst in [r.dst, *r.more_dst]:
            for k, v in enumerate(r.variants):
                w = f"{r.kind} '{r.id}'.variants[{k}]"
                tokens = _tokens(r.source, w, v) or r.tokens
                placement = _placement(r.source, w, v["placement"]) if "placement" in v else r.placement
                tp = v.get("target_poses", r.target_poses)
                jobs.append(LinkJob(r.id, n, r.kind, r.src, dst, tokens, placement, tp, r.source))
                n += 1
    return jobs


def link_plan_line(j: LinkJob, state: str = "todo") -> str:
    pl = j.placement if isinstance(j.placement, str) else "dx{dx:+.2f} dz{dz:+.2f} dyaw{dyaw:+.0f}".format(**j.placement)
    tok = str(j.tokens[0]) if len(j.tokens) == 1 else f"{j.tokens[0]}-{j.tokens[-1]}"
    return (f"{state:4}  {j.stem():40} {j.kind:9} {j.src.label()} -> {j.dst.label()}  tokens {tok} ({j.frames_label()}) "
            f"placement {pl} target_poses {j.target_poses}")
