# SPDX-License-Identifier: GPL-3.0-or-later
"""`python -m motiongen selftest`: parse every recipe in recipes/, plan them, check the format rules; `grade` gates on
synthetic BVH fixtures; hub poses (recipes/hubs.toml); MG8 link recipes and the numpy SOMA-77 port. No GPU; torch + Kimodo (CPU) only for the hub-pose constraint
smoke test, skipped when they are not importable."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import os

from .hubs import DEFAULT_HUBS, HubError, load_hubs, parse_hubs_text
from .recipe import KIMODO_FPS, MAX_KEYFRAMES, RecipeError, load_bands, load_recipes, parse_recipe_text, plan, plan_line

HERE = Path(__file__).resolve().parent

GOOD = """
[[activity]]
id = "wipe.table.cloth"
point = "table"
phases = ["enter", "loop"]
variants = 3
duration = { enter = 2.0, loop = 9.0 }
prompt = "A person wipes a table with a cloth in the right hand."
constraints.enter = { pelvis_height = { height = 0.9, at = "end" } }
constraints.loop = { root = "fixed", hand_r = { height = "band.table+0.05", forward = 0.4, side = 0.15 }, keys = 0.25 }
"""

KEYED = """
[[activity]]
id = "swing.keyed"
point = "ground"
phases = ["act"]
duration = 2.0
prompt = "A person swings."
constraints.act = { root = "path:0,1@1.0;0,2@2.0", pelvis_height = [{ height = 0.95, at = [0.0, 2.0] }, { height = 0.5, at = [0.5] }], hand_r = [{ height = 1.7, forward = 0.1, side = 0.3, at = [0.3] }, { height = 0.8, forward = 0.5, side = -0.3, at = [0.8] }] }
"""

PROFILE = """
[[activity]]
id = "walk.profile"
point = "ground"
phases = ["cycle", "start", "stop", "turn"]
duration = { cycle = 6.0, start = 4.0, stop = 4.0, turn = 3.0 }
prompt = "A person walks forward at a relaxed pace."
cfg_weight = [2.0, 3.0]
constraints.cycle = { root = "profile:speed=1.4" }
constraints.start = { root = "profile:hold=0.5,ramp=1.5,speed=1.6" }
constraints.stop = { root = "profile: speed=1.6, decel=0.9, rest=0.8" }
constraints.turn = { root = "profile:hold=0.4,ramp=1.2,speed=1.4,turn=90" }
"""

# Profile / cfg_weight syntax errors: (activity body, message needle, field the message must name).
PROFILE_BAD = {
    "profile speed not a number": ('constraints.loop = { root = "profile:speed=fast" }', "not a number", ".constraints.loop.root"),
    "profile without speed": ('constraints.loop = { root = "profile:hold=0.5" }', "needs speed", ".constraints.loop.root"),
    "profile unknown field": ('constraints.loop = { root = "profile:speed=1,jog=2" }', "bad profile field 'jog=2'", ".constraints.loop.root"),
    "profile field twice": ('constraints.loop = { root = "profile:speed=1,speed=2" }', "given twice", ".constraints.loop.root"),
    "profile negative segment": ('constraints.loop = { root = "profile:speed=1,ramp=-1" }', ">= 0", ".constraints.loop.root"),
    "profile turn without ramp": ('constraints.loop = { root = "profile:speed=1,turn=90" }', "turn needs ramp", ".constraints.loop.root"),
    "profile overruns phase": ('constraints.loop = { root = "profile:hold=2,ramp=1.5,speed=1" }', "exceed the phase duration 3 s", ".constraints.loop.root"),
    "cfg_weight not a pair": ("cfg_weight = 2.0", "expected [text, constraint]", ".cfg_weight"),
    "cfg_weight negative": ("cfg_weight = [2.0, -1.0]", "expected [text, constraint]", ".cfg_weight"),
}

# sha256 of `motiongen plan recipes/loco-batch1.toml` rows (no summary line) before AN4a: recipes without the new
# fields must plan byte-identically. Changes only if loco-batch1.toml (a shipped batch) or the plan format changes.
LOCO_BATCH1_PLAN_SHA = "2823ea6f6a3a848b46c36de4a4fece92f126df97568247f0964cf8ac59cf81ee"
# Same for recipes/loco-batch2.toml, taken before AN44a (hub poses).
LOCO_BATCH2_PLAN_SHA = "25af5074b8d179c53a2fdfc32d7cad9873a873879daa55d4b5f11bfa161ceb27"

# Hub poses (AN44a): a test hub table (sources are placeholders; the torch smoke test writes its own BVH).
HUBS_TEST = """
[[hub]]
id = "stand.idle"
family = "any"
body = "stand.idle"
source = "x/stand.bvh"
frame = 1

[[hub]]
id = "sit.chair+R:cup"
family = "chair"
body = "sit.chair"
hands = { R = "cup" }
source = "x/sit.bvh"
frame = 2
"""

POSES = """
[[activity]]
id = "sit.chair.piece"
point = "ground"
phases = ["enter", "turn", "hold"]
duration = { enter = 2.0, turn = 3.0, hold = 1.0 }
prompt = "A person sits down."
constraints.enter = { start_pose = "hub:stand.idle", end_pose = "hub:sit.chair+R:cup", root = "path:0,0.5" }
constraints.turn = { end_pose = "hub:stand.idle", root = "profile:hold=0.4,ramp=1.2,speed=1.4,turn=90" }
constraints.hold = { start_pose = "hub:sit.chair+R:cup", end_pose = "hub:sit.chair+R:cup", pose_frames = 3 }
"""

# (constraint table body, message needle, field the message must name) for one phase "loop" of 3 s.
POSE_BAD = {
    "unknown hub id": ('{ start_pose = "hub:kneel" }', "no hub 'kneel'", ".constraints.loop.start_pose"),
    "pose without hub: prefix": ('{ end_pose = "stand.idle" }', "expected 'hub:<id>'", ".constraints.loop.end_pose"),
    "pose_frames without a pose": ('{ pose_frames = 2 }', "needs start_pose or end_pose", ".constraints.loop.pose_frames"),
    "pose_frames over budget": ('{ start_pose = "hub:stand.idle", end_pose = "hub:stand.idle", pose_frames = 10 }', "20 held pose frames",
                                ".constraints.loop.pose_frames"),
}

HUBS_BAD = {
    "hub id not body + hands": ('id = "sit.chair"\nfamily = "chair"\nbody = "sit.chair"\nhands = { L = "book" }\nsource = "a.bvh"\nframe = 0',
                                "id must be body + hand state, here 'sit.chair+L:book'"),
    "hub unknown key": ('id = "kneel"\nfamily = "any"\nbody = "kneel"\nsource = "a.bvh"\nframe = 0\nangles = [1]', "unknown keys ['angles']"),
    "hub bad hand": ('id = "kneel"\nfamily = "any"\nbody = "kneel"\nhands = { R = "Cup" }\nsource = "a.bvh"\nframe = 0', "hands.R = 'Cup'"),
    "hub source not bvh": ('id = "kneel"\nfamily = "any"\nbody = "kneel"\nsource = "a.fbx"\nframe = 0', "must be a .bvh path"),
    "hub negative frame": ('id = "kneel"\nfamily = "any"\nbody = "kneel"\nsource = "a.bvh"\nframe = -1', "0-based frame"),
}

BAD = {
    "unknown activity key": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\ncolour = 1', "unknown keys ['colour']"),
    "unknown constraint key": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { pelvis = 0.5 }', "unknown keys ['pelvis']"),
    "unknown hand key": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { hand_r = { height = 1.0, reach = 2 } }', "unknown keys ['reach']"),
    "bad band ref": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { pelvis_height = "seat" }', "band.<kind>"),
    "constraints for missing phase": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.exit = { root = "fixed" }', "unknown keys ['exit']"),
    "too long": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 12.0\nprompt = "x"', "Kimodo limit"),
    "missing prompt phase": ('id = "a.b"\npoint = "seat"\nphases = ["enter", "loop"]\nduration = 3.0\nprompt.loop = "x"', "missing phases ['enter']"),
    "untimed path point among timed": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { root = "path:0,1@1;0,2" }', "every path point"),
    "hand list without at": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { hand_r = [{ height = 1.0 }] }', "needs 'at'"),
    "bad root": ('id = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { root = "pinned" }', "expected 'fixed'"),
}


# Minimal SOMA-named rig for `grade` fixtures (cm, Y up, facing +Z, left on +X): feet 7 cm (ankle) / 2 cm (toe) up.
RIG = [("Hips", None, (0, 100, 0)), ("LeftLeg", "Hips", (10, -5, 0)), ("LeftShin", "LeftLeg", (0, -45, 0)),
       ("LeftFoot", "LeftShin", (0, -43, 0)), ("LeftToeBase", "LeftFoot", (0, -5, 12)),
       ("RightLeg", "Hips", (-10, -5, 0)), ("RightShin", "RightLeg", (0, -45, 0)),
       ("RightFoot", "RightShin", (0, -43, 0)), ("RightToeBase", "RightFoot", (0, -5, 12)),
       ("Head", "Hips", (0, 60, 0)), ("LeftHand", "Hips", (20, 20, 0)), ("RightHand", "Hips", (-20, 20, 0))]


def fixture_bvh(frames: list[tuple[float, float, float, float]], fps: int = 30) -> str:
    """BVH text; each frame = Hips (x, y, z cm, yaw deg about +Y, + = left); every other joint keeps its rest pose."""
    def joint(i: int, depth: int) -> list[str]:
        name, _, off = RIG[i]
        pad = "  " * depth
        head = "ROOT" if i == 0 else "JOINT"
        ch = "6 Xposition Yposition Zposition Yrotation Xrotation Zrotation" if i == 0 else "3 Zrotation Yrotation Xrotation"
        out = [f"{pad}{head} {name}", pad + "{", f"{pad}  OFFSET {off[0]} {off[1]} {off[2]}", f"{pad}  CHANNELS {ch}"]
        for k, (_, parent, _) in enumerate(RIG):
            if parent == name:
                out += joint(k, depth + 1)
        return out + [pad + "}"]
    lines = ["HIERARCHY"] + joint(0, 0) + ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1 / fps:.6f}"]
    rest = " ".join(["0"] * 3 * (len(RIG) - 1))
    lines += [f"{x} {y} {z} {yaw} 0 0 {rest}" for x, y, z, yaw in frames]
    return "\n".join(lines) + "\n"


def write_fixture(root: Path, activity: str, take: int, frames, prov: bool = True) -> Path:
    stem = f"{activity}.act.ground0.{take:02d}"
    d = root / activity / "act"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.bvh").write_text(fixture_bvh(frames), encoding="utf-8")
    if prov:
        (d / f"{stem}.prov.json").write_text(json.dumps({"id": stem, "activity": activity, "phase": "act"}), encoding="utf-8")
    return d / f"{stem}.bvh"


def grade_checks(check) -> None:
    from .grade import grade_clip, grade_dir, write_json_atomic

    n = 60  # 2 s at 30 fps
    still = [(0.0, 100.0, 0.0, 0.0)] * n
    slide = [(0.0, 100.0, 100.0 * t / 30, 0.0) for t in range(n)]  # 1 m/s, feet planted on the floor: all skate
    pop = [(0.0, 100.0, 10.0 if t == 30 else 0.0, 0.0) for t in range(n)]  # one-frame 10 cm pelvis pop
    turn = lambda deg: [(0.0, 100.0, 0.0, deg * t / (n - 1)) for t in range(n)]  # spin on the spot, feet never lift
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _, g = grade_clip(write_fixture(root, "t.walk.cycle.still", 0, still))
        m = g["metrics"]
        check(g["type"] == "cycle" and g["gates"] == [] and m["skate_cm_s"] == 0.0 and m["jerk_p99"] == 0.0 and m["steps"] == 0,
              f"grade: still cycle has no gate, zero skate/jerk: {m}")
        _, g = grade_clip(write_fixture(root, "t.walk.cycle.slide", 0, slide))
        # 59 sliding steps of 3.33 cm per foot over 60 contact frames: 100 * 59 / 60 = 98.3 cm/s
        check(g["gates"] == ["skate"] and g["metrics"]["skate_cm_s"] == 98.3,
              f"grade: sliding feet at 1 m/s -> skate gate ({g['metrics']['skate_cm_s']} cm/s)")
        _, g = grade_clip(write_fixture(root, "t.walk.cycle.pop", 0, pop))
        check("jerk" in g["gates"], f"grade: one-frame pop -> jerk gate (p99 {g['metrics']['jerk_p99']} m/s^3)")
        _, g = grade_clip(write_fixture(root, "t.turn.l90.a", 0, turn(90.0)))
        m = g["metrics"]
        check(g["type"] == "turn" and g["gates"] == ["no_step"] and not g["mirror_candidate"]
              and m["turn_deg"] == 90.0 and m["turn_err_deg"] == 0.0 and m["turn_target_deg"] == 90.0,
              f"grade: turn-in-place l90 on planted feet -> no_step only: {g['gates']} {m}")
        _, g = grade_clip(write_fixture(root, "t.turn.l90.b", 0, turn(-90.0)))
        check(g["gates"] == ["no_step"] and g["mirror_candidate"] and g["metrics"]["turn_deg"] == -90.0,
              "grade: l90 prompt turning right -> mirror_candidate, not a gate")
        _, g = grade_clip(write_fixture(root, "t.turn.r90.c", 0, turn(-90.0)))
        check(not g["mirror_candidate"] and g["metrics"]["turn_target_deg"] == -90.0, "grade: r90 turning right is the right way")
        _, g = grade_clip(write_fixture(root, "t.turn.l90.d", 0, turn(45.0)))
        check(g["gates"] == ["turn_mag", "no_step"] and g["metrics"]["turn_err_deg"] == 45.0 and not g["mirror_candidate"],
              f"grade: l90 prompt turning 45 deg -> turn_mag ({g['metrics']['turn_err_deg']} deg off)")
        _, g = grade_clip(write_fixture(root, "t.walk.cycle.noprov", 0, slide, prov=False))
        check(g["type"] == "cycle" and g["gates"] == ["skate"], "grade: no .prov.json -> type from the file stem")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for k in range(6):  # 4 of 6 gated -> failing; 3 of 6 -> not
            write_fixture(root, "g.run.cycle.a", k, slide if k < 4 else still)
            write_fixture(root, "g.run.cycle.b", k, slide if k < 3 else still)
        grades = grade_dir(root)
        out = root / "grades.json"
        write_json_atomic(out, grades)
        back = json.loads(out.read_text(encoding="utf-8"))
        grp = back["_groups"]
        check(grp == {"g.run.cycle.a.act.ground0": {"failing": True, "gated": 4, "total": 6},
                      "g.run.cycle.b.act.ground0": {"failing": False, "gated": 3, "total": 6}},
              f"grade: group triage 4/6 failing, 3/6 not: {grp}")
        clip = back["g.run.cycle.a.act.ground0.00"]
        check(len(back) == 13 and set(clip) == {"type", "gates", "mirror_candidate", "metrics", "source"} and clip["source"] == "py"
              and list(back) == sorted(back) and not (root / "grades.json.tmp").exists(),
              "grade: grades.json = 12 clips + _groups, sorted keys, per-clip fields, no temp file left")



HUBCHECK_HUBS = """[[hub]]
id = "stand.idle"
family = "any"
body = "stand.idle"
source = "hubs/hub.bvh"
frame = 1

[[hub]]
id = "sit.chair"
family = "chair"
body = "sit.chair"
source = "hubs/hub.bvh"
frame = 2
"""


def rot_fixture(frames, rots=None) -> str:
    """`fixture_bvh` plus local rotations: rots[t] = {joint: (z, y, x) deg} on frame t (RIG order = BVH order)."""
    lines = fixture_bvh(frames).split("\n")
    first = next(i for i, ln in enumerate(lines) if ln.startswith("Frame Time")) + 1
    names = [n for n, _, _ in RIG]
    for t, r in enumerate(rots or []):
        tok = lines[first + t].split()
        for name, zyx in r.items():
            base = 6 + 3 * (names.index(name) - 1)
            tok[base:base + 3] = [str(v) for v in zyx]
        lines[first + t] = " ".join(tok)
    return "\n".join(lines)


def hubcheck_checks(check) -> None:
    from .grade import write_json_atomic
    from .hubcheck import hubcheck_dir, table

    still = [(0.0, 100.0, 0.0, 0.0)] * 10
    hubs = parse_hubs_text(HUBCHECK_HUBS, "<hubs>")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "hubs").mkdir()
        (root / "hubs" / "hub.bvh").write_text(fixture_bvh(still[:3]), encoding="utf-8")
        takes = root / "takes"
        takes.mkdir()

        def take(name: str, text: str, prov: dict) -> None:
            (takes / f"{name}.bvh").write_text(text, encoding="utf-8")
            (takes / f"{name}.prov.json").write_text(json.dumps({"id": name, **prov}), encoding="utf-8")

        both = {"from_hub": "stand.idle", "to_hub": "sit.chair"}
        take("h.same.act.any.00", fixture_bvh(still), both)
        take("h.shin.act.any.00", rot_fixture(still, [{}] * 9 + [{"LeftShin": (0, 0, 30)}]), both)
        moved = [(50.0, 100.0, -20.0, 70.0)] * 5 + [(50.0, 90.0, -20.0, 70.0)] * 5
        take("h.moved.act.any.00", fixture_bvh(moved), both)
        take("h.slide.act.any.00", fixture_bvh([(0.0, 100.0, 100.0 * t / 30, 0.0) for t in range(30)]), {"from_hub": "stand.idle"})
        take("h.plain.act.any.00", fixture_bvh(still), {"activity": "h.plain"})
        res = hubcheck_dir(takes, hubs, root)
        out = takes / "hubcheck.json"
        write_json_atomic(out, res)
        back = json.loads(out.read_text(encoding="utf-8"))

        s = back["h.same.act.any.00"]
        zero = all(s[e]["rot_max_deg"] == 0.0 and s[e]["worst_cm"] == 0.0 and s[e]["pelvis_dy_cm"] == 0.0 for e in ("start", "end"))
        check(zero and s["start"]["joints"] == len(RIG) and s["slide_mean_cm_s"] == 0.0,
              f"hubcheck: a take equal to the hub frames scores 0 deg / 0 cm / 0 slide ({s['start']['joints']} joints)")
        sh = back["h.shin.act.any.00"]
        e = sh["end"]
        knee = 2 * 43 * math.sin(math.radians(15))  # 43 cm shin swung 30 deg: chord at the ankle
        check(sh["start"]["rot_max_deg"] == 0.0 and e["rot_max_deg"] == 30.0 and e["rot_worst_joint"] == "LeftShin"
              and e["rot_mean_deg"] == round(30.0 / len(RIG), 2) and e["effector_cm"]["foot_l"] == round(knee, 1)
              and e["worst_cm"] == e["effector_cm"]["foot_l"] and e["effector_cm"]["foot_r"] == 0.0,
              f"hubcheck: 30 deg on the left shin at the end -> 30 deg max, left foot {round(knee, 1)} cm off: {e}")
        mv = back["h.moved.act.any.00"]
        st, en = mv["start"], mv["end"]
        check(st["rot_max_deg"] == 0.0 and st["worst_cm"] == 0.0 and en["rot_max_deg"] == 0.0
              and en["pelvis_dy_cm"] == -10.0 and en["effector_cm"]["pelvis"] == 10.0,
              f"hubcheck: pelvis XZ (0.5, -0.2) m + 70 deg heading align away; a 10 cm drop shows (dy {en['pelvis_dy_cm']} cm)")
        sl = back["h.slide.act.any.00"]
        check(sl["slide_mean_cm_s"] == 100.0 and sl["slide_p95_cm_s"] == 100.0 and "end" not in sl and sl["start"]["worst_cm"] == 0.0,
              f"hubcheck: planted feet sliding at 1 m/s -> 100 cm/s mean and p95; no to_hub, no end check: {sl}")
        check(back["_skipped"] == ["h.plain.act.any.00"] and set(back) - {"_groups", "_skipped"} == {
              "h.same.act.any.00", "h.shin.act.any.00", "h.moved.act.any.00", "h.slide.act.any.00"}
              and back["_groups"]["h.shin.act.any"]["takes"] == 1 and not (takes / "hubcheck.json.tmp").exists(),
              f"hubcheck: take without from_hub / to_hub skipped; groups + atomic json: {back['_skipped']}")
        rows = table(res)
        check(len(rows) == 2 + 4 + 3 + 4, f"hubcheck: table = 4 take rows + 4 group rows ({len(rows)} lines)")

def profile_checks(check, bands) -> None:
    jobs = {j.phase: j for j in plan(parse_recipe_text(PROFILE, "<profile>"), bands)}
    dense = all(j.constraints["root"]["frames"] == list(range(j.num_frames)) and len(j.constraints["root"]["points"]) == j.num_frames
                and j.constraints["root"]["type"] == "path" for j in jobs.values())
    check(dense, "profile: one root key per output frame, type 'path' (all four phases)")

    def pts(phase):
        return jobs[phase].constraints["root"]["points"]

    def steps(phase):  # per-frame displacement along the path
        p = pts(phase)
        return [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])]

    near = lambda a, b, tol=2e-4: abs(a - b) <= tol  # noqa: E731 (points are rounded to 0.1 mm)
    c = pts("cycle")
    t_last = (jobs["cycle"].num_frames - 1) / KIMODO_FPS
    check(near(c[-1][1], 1.4 * t_last) and all(near(d, 1.4 / KIMODO_FPS) for d in steps("cycle")) and all(p[0] == 0.0 for p in c),
          f"profile cycle: steady 1.4 m/s along +Z, {c[-1][1]} m at frame {len(c) - 1} (closed form {1.4 * t_last:.4f})")

    s = pts("start")
    hold_f = int(0.5 * KIMODO_FPS)
    ramp_end = int(2.0 * KIMODO_FPS)
    t_last = (jobs["start"].num_frames - 1) / KIMODO_FPS
    want = 1.6 * (1.5 / 2 + (t_last - 2.0))  # ramp covers half its time at full speed, then steady to the last frame
    ds = steps("start")
    check(all(p == [0.0, 0.0] for p in s[: hold_f + 1]) and near(s[-1][1], want)
          and all(b >= a - 1e-4 for a, b in zip(ds[hold_f:ramp_end], ds[hold_f + 1 : ramp_end])) and near(ds[-1], 1.6 / KIMODO_FPS),
          f"profile start: still for the 0.5 s hold, speed rises smoothly, ends at 1.6 m/s, {s[-1][1]} m (closed form {want:.4f})")

    st = pts("stop")
    want = 1.6 * ((4.0 - 1.7) + 0.9 / 2)  # steady 2.3 s, then half the decel time at full speed
    still_f = math.ceil(3.2 * KIMODO_FPS)
    ds = steps("stop")
    check(all(p == st[-1] for p in st[still_f:]) and near(st[-1][1], want) and near(ds[0], 1.6 / KIMODO_FPS)
          and all(b <= a + 1e-4 for a, b in zip(ds[69:still_f], ds[70:still_f])),
          f"profile stop: steady 1.6 m/s, smooth decel, zero speed in the 0.8 s rest, {st[-1][1]} m total (closed form {want:.4f})")

    tr = jobs["turn"].constraints["root"]
    arc = sum(steps("turn"))
    t_last = (jobs["turn"].num_frames - 1) / KIMODO_FPS
    want = 1.4 * (1.2 / 2 + (t_last - 1.6))
    last = [tr["points"][-1][0] - tr["points"][-2][0], tr["points"][-1][1] - tr["points"][-2][1]]
    check(tr["heading"] == {"frames": [jobs["turn"].num_frames - 1], "cos_sin": [[0.0, 1.0]]} and near(arc, want, 5e-3)
          and last[0] > 0 and abs(last[1]) < 1e-4 and tr["points"][hold_f - 3] == [0.0, 0.0],
          f"profile turn=90: travels +X at the end (left), heading key [cos, sin] = [0, 1] at the last frame, arc {arc:.4f} m ~ {want:.4f}")

    cyc = jobs["cycle"]
    check(cyc.cfg_weight == [2.0, 3.0] and " cfg [2.0,3.0] " in plan_line(cyc),
          "cfg_weight = [text, constraint] reaches the job and the plan row")
    from .__main__ import _cfg_prov

    class FakeWorker:  # stands in for KimodoWorker.cfg_default (reads Kimodo's signature at run time)
        def cfg_default(self):
            return {"cfg_type": "separated", "cfg_weight": [2.0, 2.0]}

    other = next(iter(plan(parse_recipe_text(GOOD, "<good>"), bands)))
    check(_cfg_prov(cyc, FakeWorker()) == {"cfg_type": "separated", "cfg_weight": [2.0, 3.0], "cfg_weight_source": "recipe"}
          and _cfg_prov(other, FakeWorker())["cfg_weight_source"] == "kimodo default",
          ".prov.json guidance: the recipe's cfg_weight, else Kimodo's default (marked as such)")

    for label, (extra, needle, field_) in PROFILE_BAD.items():
        body = 'id = "a.b"\npoint = "ground"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\n' + extra
        try:
            parse_recipe_text("[[activity]]\n" + body + "\n", "<bad>")
            check(False, f"rejects {label}")
        except RecipeError as e:
            m = str(e)
            check(needle in m and m.startswith("<bad>: ") and f"activity 'a.b'{field_}" in m, f"rejects {label}: {m}")


def hub_checks(check, bands) -> None:
    real = load_hubs(DEFAULT_HUBS)
    check({"stand.idle", "sit.chair", "sit.stool", "sit.chair.seat54", "sit.bench", "sit.floor", "sit.chair@old", "stand.idle@old"} == set(real) and real["sit.chair"].family == "chair" and real["sit.chair@old"].body == "sit.chair@old"
          and real["stand.idle"].hands == {"L": "empty", "R": "empty"} and all(h.source.endswith(".bvh") for h in real.values()),
          f"recipes/hubs.toml: {len(real)} hubs parse ({', '.join(sorted(real))})")
    root = os.environ.get("MOTIONGEN_HUB_ROOT")
    if root:  # the seeded sources exist (only where the generated takes are, e.g. 'animation packs/generated')
        missing = [h.id for h in real.values() if not h.source_path(Path(root)).exists()]
        check(not missing, f"hubs.toml sources exist under MOTIONGEN_HUB_ROOT ({root}); missing: {missing}")
    for label, (body, needle) in HUBS_BAD.items():
        try:
            parse_hubs_text("[[hub]]\n" + body + "\n", "<hubs>")
            check(False, f"rejects {label}")
        except HubError as e:
            check(needle in str(e) and str(e).startswith("<hubs>: hub"), f"rejects {label}: {e}")
    try:
        load_recipes([DEFAULT_HUBS])
        check(False, "hubs.toml passed as a recipe is an error")
    except RecipeError as e:
        check("a hubs file" in str(e), f"hubs.toml passed as a recipe is an error: {e}")

    hubs = parse_hubs_text(HUBS_TEST, "<hubs>")
    jobs = {j.phase: j for j in plan(parse_recipe_text(POSES, "<poses>", hubs), bands)}
    en, tu, ho = jobs["enter"], jobs["turn"], jobs["hold"]
    s0, e0 = en.constraints["start_pose"], en.constraints["end_pose"]
    check(s0 == {"hub": "stand.idle", "family": "any", "source": "x/stand.bvh", "source_frame": 1, "frames": [0],
                 "root_xz": [[0.0, 0.0]], "heading_cos_sin": [1.0, 0.0]}
          and e0["hub"] == "sit.chair+R:cup" and e0["frames"] == [en.num_frames - 1] and e0["root_xz"] == [[0.0, 0.5]]
          and e0["heading_cos_sin"] == [1.0, 0.0] and e0["source_frame"] == 2,
          f"start_pose at frame 0 (origin, +Z), end_pose at the last frame on the root path's end: {e0['frames']} {e0['root_xz']}")
    te = tu.constraints["end_pose"]
    rt = tu.constraints["root"]
    check(te["heading_cos_sin"] == [0.0, 1.0] and te["root_xz"] == [rt["points"][-1]] and "start_pose" not in tu.constraints,
          f"end_pose on a turning profile: the profile's end point {te['root_xz']} and final heading {te['heading_cos_sin']}")
    hs, he = ho.constraints["start_pose"], ho.constraints["end_pose"]
    check(hs["frames"] == [0, 1, 2] and he["frames"] == [27, 28, 29] and hs["root_xz"] == [[0.0, 0.0]] * 3,
          f"pose_frames = 3 holds the pose on frames {hs['frames']} / {he['frames']}")
    row = plan_line(en)
    check(" hubs stand.idle->sit.chair+R:cup {" in row and " hubs none->stand.idle {" in plan_line(tu),
          "plan rows print 'hubs a->b' ('none' for an unset end)")
    from .__main__ import _hub_prov

    check(_hub_prov(en) == {"from_hub": "stand.idle", "to_hub": "sit.chair+R:cup"} and _hub_prov(tu) == {"to_hub": "stand.idle"},
          ".prov.json: from_hub / to_hub when set (the hub frames ride in 'constraints')")

    for label, (table, needle, field_) in POSE_BAD.items():
        body = f'id = "a.b"\npoint = "ground"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = {table}'
        try:
            parse_recipe_text("[[activity]]\n" + body + "\n", "<bad>", hubs)
            check(False, f"rejects {label}")
        except RecipeError as e:
            m = str(e)
            check(needle in m and m.startswith("<bad>: ") and f"activity 'a.b'{field_}" in m, f"rejects {label}: {m}")
    clash = ('[[activity]]\nid = "a.b"\npoint = "ground"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\n'
             'constraints.loop = { end_pose = "hub:stand.idle", pelvis_height = { height = 0.9, at = "end" } }\n')
    try:
        plan(parse_recipe_text(clash, "<bad>", hubs), bands)
        check(False, "rejects a pelvis_height key on a hub pose frame")
    except RecipeError as e:
        check("<bad>: activity 'a.b'.constraints.loop" in str(e) and "end_pose: frames [89] are also keyed by pelvis_height" in str(e),
              f"rejects a pelvis_height key on a hub pose frame: {e}")

    hub_pose_smoke(check, bands)


def link_checks(check) -> None:
    """MG8 link recipes, the sample-pose refusal, the numpy SOMA-77 port (BVH round trip) and link placement."""
    import numpy as np

    from . import soma
    from .clipstats import _rot
    from .hubcheck import hubcheck_dir, table
    from .link_worker import _feet, place_target
    from .links import link_plan_line, load_links, parse_links_text, plan_links

    recipe = HERE / "recipes" / "links-spike.toml"
    jobs = plan_links(load_links([recipe]), Path("gen"))
    by = {j.stem(): j for j in jobs}
    check(sorted(by) == ["link.seat.chair.enter.00", "link.seat.chair.enter.01", "link.seat.chair.exit.00", "seam.sit.chair.rest.00"]
          and by["link.seat.chair.enter.00"].tokens == tuple(range(8, 15)) and by["link.seat.chair.enter.01"].tokens == tuple(range(6, 17))
          and by["seam.sit.chair.rest.00"].tokens == (6,) and by["seam.sit.chair.rest.00"].placement == "seam"
          and by["link.seat.chair.exit.00"].src.hub == "sit.chair" and by["link.seat.chair.exit.00"].dst.hub == "stand.idle",
          f"links-spike.toml: {len(jobs)} link takes plan (enter x2 token ranges, exit, loop seam)")
    sit = plan_links(load_links([HERE / "recipes" / "sit-batch1-links.toml"]), Path("gen"))
    kinds = {k: sum(j.kind == k for j in sit) for k in ("link", "hub_loop")}
    check(kinds["link"] == 96 and kinds["hub_loop"] >= 6,
          f"sit-batch1-links.toml (AN26a): {kinds['link']} enter/exit takes + {kinds['hub_loop']} hub loops plan")
    row = link_plan_line(by["link.seat.chair.enter.00"])
    check(" hub:stand.idle -> hub:sit.chair  tokens 8-14 (26-50f) placement feet target_poses 4" in row, f"link plan line: {row.strip()}")
    seam = by["seam.sit.chair.rest.00"]
    check(seam.src.at == "end" and seam.dst.at == "start" and seam.src.path == seam.dst.path and seam.frames_label() == "18f",
          "loop seam: clip end -> its own start, 6 tokens = 18 frames")
    one = parse_links_text('[[link]]\nid = "a.b"\nfrom = "clip:x/a.bvh#end"\nto = "clip:x/b.bvh#12"\nseconds = 1.0\n'
                           'placement = { dx = 0.1, dyaw = 90 }\n', "<l>")[0]
    check(one.tokens == (9,) and one.dst.at == "12" and one.placement == {"dx": 0.1, "dz": 0.0, "dyaw": 90.0},
          "seconds = 1.0 -> 9 tokens (30 frames); clip frame index; explicit placement")
    head = '[[link]]\nid = "a.b"\n'
    hubs2 = 'from = "hub:stand.idle"\nto = "hub:sit.chair"\n'
    bad = {
        "tokens out of range": (head + hubs2 + "tokens = 5\n", "tokens = 5"),
        "tokens and seconds": (head + hubs2 + "tokens = 8\nseconds = 1.0\n", "not both"),
        "a bad pose ref": (head + 'from = "stand.idle"\nto = "hub:sit.chair"\n', "expected 'hub:<id>'"),
        "an unknown hub": (head + 'from = "hub:fly"\nto = "hub:sit.chair"\n', "no hub 'fly'"),
        "a bad placement": (head + hubs2 + 'placement = "air"\n', "placement = 'air'"),
        "a seam placement": ('[[loop_seam]]\nid = "a.b"\nclip = "x.bvh"\nvariants = [{ placement = "feet" }]\n', "placement is fixed"),
        "an unknown key": (head + hubs2 + "seed = 3\n", "unknown keys ['seed']"),
    }
    for label, (text, needle) in bad.items():
        try:
            parse_links_text(text, "<bad>")
            check(False, f"link recipe rejects {label}")
        except RecipeError as e:
            check(needle in str(e), f"link recipe rejects {label}: {e}")
    try:
        parse_recipe_text(recipe.read_text(encoding="utf-8"), str(recipe))
        check(False, "plan/run refuse a link recipe")
    except RecipeError as e:
        check("a link recipe" in str(e), "plan/run refuse a link recipe")

    with tempfile.TemporaryDirectory() as d:
        mb = Path(d) / "motionbricks"
        sample = mb / "assets" / "motions" / "soma" / "sitting" / "sit_on_chair_loop.bvh"
        old = os.environ.get("MOTIONGEN_MOTIONBRICKS")
        os.environ["MOTIONGEN_MOTIONBRICKS"] = str(mb)
        try:
            cases = (("MotionBricks sample folder", sample), ("BONES take name", Path(d) / "neutral_walk_ff_180_R_002__A536_M.bvh"))
            for label, path in cases:
                for kind, text in (("link", head + f'from = "hub:stand.idle"\nto = "clip:{path.as_posix()}#start"\n'),
                                   ("loop_seam", f'[[loop_seam]]\nid = "a.b"\nclip = "{path.as_posix()}"\n')):
                    try:
                        plan_links(parse_links_text(text, "<refuse>"), Path(d))
                        check(False, f"refuses a {label} as a {kind} pose source")
                    except RecipeError as e:
                        check("evaluation" in str(e).lower(), f"refuses a {label} as a {kind} pose source")
            hub = parse_hubs_text('[[hub]]\nid = "stand.idle"\nfamily = "any"\nbody = "stand.idle"\n'
                                  f'source = "{sample.as_posix()}"\nframe = 0\n', "<h>")
            try:
                plan_links(parse_links_text(head + 'from = "hub:stand.idle"\nto = "hub:stand.idle"\n', "<refuse>", hub), Path(d))
                check(False, "refuses a hub sourced from a sample motion")
            except RecipeError as e:
                check("evaluation" in str(e).lower(), f"refuses a hub sourced from a sample motion: {str(e)[:110]}...")
        finally:
            if old is None:
                os.environ.pop("MOTIONGEN_MOTIONBRICKS", None)
            else:
                os.environ["MOTIONGEN_MOTIONBRICKS"] = old

        # numpy SOMA-77 port: random poses through both rest conventions and the BVH writer / reader
        rng = np.random.default_rng(7)
        T = 6
        q = rng.normal(size=(T, 77, 4))
        q[..., 3] += 4.0  # moderate rotations
        local = soma.quat_to_mat(q)
        root = rng.normal(size=(T, 3)) * 0.3 + [0.0, 0.95, 0.0]
        check(np.abs(soma.quat_to_mat(soma.mat_to_quat(local)) - local).max() < 1e-9, "quat <-> matrix round trip")
        check(np.abs(soma.to_standard_tpose(soma.from_standard_tpose(local)) - local).max() < 1e-9,
              "T-pose <-> BONES-SEED rest round trip")
        g = soma.rot_y(0.4) @ np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        gz = np.array([[0.0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]) @ np.array([[1.0, 0, 0], [0, 0.6, -0.8], [0, 0.8, 0.6]])
        e = soma.mat_to_euler_zyx_deg(np.stack([g, gz]))
        back = [_rot("Z", np.array(z)) @ _rot("Y", np.array(y)) @ _rot("X", np.array(x)) for z, y, x in e]
        check(max(np.abs(b - m).max() for b, m in zip(back, (g, gz))) < 1e-9, "ZYX Euler matches the BVH reader (incl. gimbal lock)")
        errs = {}
        for kind in ("tpose", "seed"):
            p = Path(d) / f"rt_{kind}.bvh"
            soma.write_bvh(p, local, root, 30.0, kind)
            l2, r2, k2, fps = soma.read_soma_bvh(p)
            errs[kind] = (k2, float(soma.angle_deg(l2, local).max()), float(np.abs(r2 - root).max()), fps)
        check(all(errs[k][0] == k and errs[k][1] < 1e-2 and errs[k][2] < 1e-7 and errs[k][3] == 30.0 for k in errs),
              f"BVH round trip, both rests detected: max deg / m {[(k, round(v[1], 5), v[2]) for k, v in errs.items()]}")

        # placement: target = a turned, shifted copy; `feet` puts its heading and ankles on the context's
        rest_l = np.broadcast_to(np.eye(3), (4, 77, 3, 3)).copy()
        rest_r = np.tile([0.0, 0.95, 0.0], (4, 1))
        tl = rest_l.copy()
        tl[:, 0] = soma.rot_y(1.2) @ tl[:, 0]
        tr = rest_r + [2.0, 0.0, -1.0]
        pl, pr = place_target((rest_l, rest_r), (tl, tr), "feet")
        (h0, a0), (h1, a1) = _feet(rest_l[-1], rest_r[-1]), _feet(pl[0], pr[0])
        check(abs(h0 - h1) < 1e-9 and np.abs(a0 - a1).max() < 1e-9, "placement 'feet': target heading and ankle XZ = the context's")
        pl, pr = place_target((rest_l, rest_r), (tl, tr), {"dx": 0.0, "dz": 0.5, "dyaw": 0.0})
        fwd = soma.rot_y(h0) @ np.array([0.0, 0.0, 0.5])
        check(np.abs(_feet(pl[0], pr[0])[1] - (a0 + fwd[[0, 2]])).max() < 1e-9, "placement dz = 0.5 m forward of the context")

        # hubcheck on a link take whose ends are clip frames (+ the correction table)
        clip = Path(d) / "clip.bvh"
        soma.write_bvh(clip, local, root, 30.0)
        out = Path(d) / "links" / "x"
        out.mkdir(parents=True)
        soma.write_bvh(out / "x.00.bvh", local[2:5], root[2:5], 30.0)
        corr = {"start": {"rot_max_deg": 4.5, "joint": "RightHand", "root_cm": 1.0},
                "end": {"rot_max_deg": 2.0, "joint": "Head", "root_cm": 0.5}}
        (out / "x.00.prov.json").write_text(json.dumps({"from_clip": {"path": str(clip), "frame": 2},
                                                       "to_clip": {"path": str(clip), "frame": 4},
                                                       "correction": corr, "tokens": 6, "generate_ms": 7.0}))
        res = hubcheck_dir(Path(d) / "links", {}, Path(d))
        r = res.get("x.00", {})
        rows = table(res)
        check(r.get("start", {}).get("rot_max_deg") == 0.0 and r.get("end", {}).get("worst_cm") == 0.0
              and r.get("from_hub") == "clip.bvh#2" and any("4.5 RightHand / 1.0" in x for x in rows),
              "hubcheck: link ends on clip frames measured (0 deg / 0 cm), correction table row")

        hand_checks(check, Path(d), clip)


class FakeLinkWorker:
    """CPU stand-in for `MotionBricksWorker` (MG8b selftest): poses stay T-pose locals (p xyz at the root, q xyzw);
    a 'generation' = seeded random rotations per frame and joint, the corrected one with its ends set to the last
    context pose and the first target pose (all joints, like MotionBricks' endpoint correction)."""

    provenance = {"worker": "fake"}

    def to_mb(self, local, root):
        import numpy as np

        from . import soma
        p = np.zeros(local.shape[:2] + (3,))
        p[:, 0] = root
        return np.concatenate([p, soma.mat_to_quat(local)], -1)

    def from_mb(self, poses):
        from . import soma
        return soma.quat_to_mat(poses[..., 3:]), poses[:, 0, :3].copy()

    def generate(self, ctx, tgt, tokens, correct):
        import numpy as np

        n = 4 * tokens[0] - 6
        rng = np.random.default_rng(11)
        q = rng.normal(size=(n, ctx.shape[1], 4))
        q[..., 3] += 3.0
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        poses = np.concatenate([np.zeros((n, ctx.shape[1], 3)), q], -1)
        poses[:, 0, :3] = np.linspace(ctx[-1, 0, :3], tgt[0, 0, :3], n)
        if correct:
            poses[0], poses[-1] = ctx[-1], tgt[0]
            self.corrected = poses.copy()
        return poses, np.zeros((n, 4)), 1.0


def _slerp_reference(a, b, w):
    """Independent rotation slerp a @ exp(w log(a^T b)) by matrix log + Rodrigues (selftest reference)."""
    import numpy as np

    r = a.T @ b
    th = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0)))
    if th < 1e-12:
        return a.copy()
    k = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(th))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    t = w * th
    return a @ (np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * K @ K)


def hand_checks(check, d: Path, clip: Path) -> None:
    """MG8b hand step on a synthetic piece: wrists + fingers = the slerp, body untouched, ends exact, `model` = MG8."""
    import numpy as np

    from . import soma
    from .hubcheck import hubcheck_dir, table
    from .link_worker import WRISTS, Sources, apply_hand_mode, hand_joints, run_job, slerp_hands, smoothstep
    from .links import parse_links_text, plan_links

    r = soma.rest()
    hj = hand_joints()
    names = {r["names"][i] for i in hj}
    check(len(hj) == 50 and set(WRISTS) <= names and "LeftHandPinkyEnd" in names and "LeftForeArm" not in names,
          f"hand joints from the SOMA-77 tree: {len(hj)} (2 wrists + 2 x 24 finger joints), no forearm")
    rng = np.random.default_rng(5)
    q = rng.normal(size=(20, 77, 4))
    q[..., 3] += 2.0
    piece = soma.quat_to_mat(q)
    a, b = piece[0], piece[-1]
    out = slerp_hands(piece, a, b)
    w = smoothstep(np.arange(20) / 19)
    ref_err = max(np.abs(out[i, j] - _slerp_reference(a[j], b[j], w[i])).max() for i in range(20) for j in hj)
    body = [i for i in range(77) if i not in set(hj)]
    check(ref_err < 1e-9 and np.array_equal(out[:, body], piece[:, body])
          and np.array_equal(out[0, hj], a[hj]) and np.array_equal(out[-1, hj], b[hj]),
          f"slerp_hands: hand frames = reference slerp ({ref_err:.1e}), body joints untouched, ends exact")
    check(apply_hand_mode(piece, a, b, "model") is piece, "hand_mode 'model' leaves the model's hands (MG8 output)")

    # the worker step end to end (fake engine): BVH body = the corrected model output, hands = the slerp
    job = plan_links(parse_links_text(f'[[link]]\nid = "t.hand"\nfrom = "clip:{clip.as_posix()}#end"\n'
                                      f'to = "clip:{clip.as_posix()}#start"\ntokens = 6\nplacement = "root"\n', "<t>"), d)[0]
    src = Sources(d)
    worker = FakeLinkWorker()
    res = {}
    for mode in ("model", "slerp"):
        root_out = d / f"hand_{mode}"
        run_job(job, worker, src, root_out, mode)
        bvh, prov = job.outputs(root_out)
        got = soma.read_soma_bvh(bvh)[0]
        model_local = worker.from_mb(worker.corrected)[0]
        res[mode] = (got, model_local, json.loads(prov.read_text(encoding="utf-8")))
    got, model_local, prov_m = res["model"]
    check(float(soma.angle_deg(got, model_local).max()) < 1e-3 and prov_m["hand_mode"] == "model"
          and prov_m["correction"]["joints"] == "body",
          "run_job hand_mode 'model': BVH = the corrected model output on every joint (MG8)")
    got, model_local, prov_s = res["slerp"]
    sl = soma.read_soma_bvh(clip)[0]
    want = slerp_hands(model_local, sl[-1], sl[0])
    c = prov_s["correction"]
    check(float(soma.angle_deg(got, want).max()) < 1e-3 and float(soma.angle_deg(got[:, body], model_local[:, body]).max()) < 1e-3
          and prov_s["hand_mode"] == "slerp" and prov_s["hand_slerp"]["joints"] == 50
          and c["start"]["joint"] not in WRISTS and c["end"]["joint"] not in WRISTS and "LeftHand" in c["per_joint"]["end"],
          "run_job hand_mode 'slerp': hands = slerp(from, to), body = the corrected model output; wrists left out of "
          "the correction size, kept in per_joint (.prov.json)")
    hc = hubcheck_dir(d / "hand_slerp", {}, d)
    rr = hc.get("t.hand.00", {})
    rows = table(hc)
    check(rr.get("start", {}).get("rot_max_deg") == 0.0 and rr.get("end", {}).get("rot_max_deg") == 0.0
          and any(x.startswith("| t.hand.00 | slerp |") for x in rows) and rr["link"]["wrist_deg_s"]["l"] > 0,
          f"hubcheck on the slerp take: ends 0/0 deg, link row with hand mode + wrist speed {rr.get('link', {}).get('wrist_deg_s')}")

    # AN26a hub loop (fake engine): link hub -> clip start + the clip + link clip end -> the same hub, one take
    from .hubs import Hub
    from .link_worker import run_hub_loop
    loop_clip = d / "loop_clip.bvh"
    ll = soma.quat_to_mat(rng.normal(size=(12, 77, 4)) + [0.0, 0.0, 0.0, 3.0])
    lr = np.tile([1.5, 0.6, -2.0], (12, 1)) + np.linspace(0, 0.05, 12)[:, None]  # elsewhere, drifting 5 cm
    soma.write_bvh(loop_clip, ll, lr, 30.0)
    hub = Hub("sit.test", "test", "sit.test", {"L": "empty", "R": "empty"}, clip.as_posix(), 3, "<t>")
    hl, hr = soma.read_soma_bvh(clip)[:2]
    job = plan_links(parse_links_text(f'[[hub_loop]]\nid = "t.loop"\nhub = "sit.test"\nclip = "{loop_clip.as_posix()}"\n'
                                      'tokens = 6\n', "<t>", {"sit.test": hub}), d)[0]
    run_hub_loop(job, FakeLinkWorker(), src, d / "hub_loop", "slerp")
    bvh, prov = job.outputs(d / "hub_loop")
    jl, jr = soma.read_soma_bvh(bvh)[:2]
    pv = json.loads(prov.read_text(encoding="utf-8"))
    ends = max(float(soma.angle_deg(jl[i], hl[3]).max()) for i in (0, -1))
    check(job.kind == "hub_loop" and len(jl) == 18 + 11 + 17 and ends < 1e-2
          and np.abs(jr[0] - hr[3]).max() < 1e-6 and np.abs(jr[-1] - hr[3]).max() < 1e-6
          and pv["from_hub"] == pv["to_hub"] == "sit.test" and pv["parts"]["clip"] == [17, 28],
          f"hub_loop: 18 + 11 + 17 frames joined, both ends = the hub pose at the hub's root ({ends:.1e} deg), "
          "from_hub = to_hub")
    hc = hubcheck_dir(d / "hub_loop", {"sit.test": hub}, d).get("t.loop.00", {})
    check(hc.get("start", {}).get("rot_max_deg", 1) < 1e-2 and hc.get("end", {}).get("rot_max_deg", 1) < 1e-2,
          f"hubcheck on the hub loop: ends {hc.get('start', {}).get('rot_max_deg')}/{hc.get('end', {}).get('rot_max_deg')} deg")
    many = plan_links(parse_links_text('[[hub_loop]]\nid = "a.b"\nhub = "sit.test"\nclip = ["x/a.bvh", "x/b.bvh"]\n'
                                       'variants = [{}, { tokens = 8 }]\n', "<t>", {"sit.test": hub}), d)
    check([(j.stem(), Path(j.dst.path).name, j.tokens[0]) for j in many]
          == [("a.b.00", "a.bvh", 6), ("a.b.01", "a.bvh", 8), ("a.b.02", "b.bvh", 6), ("a.b.03", "b.bvh", 8)],
          "hub_loop clip list: one take per clip x variant, numbered in order (one prompt group)")
    try:
        parse_links_text('[[hub_loop]]\nid = "a.b"\nhub = "sit.test"\nclip = "x.bvh"\nvariants = [{ placement = "feet" }]\n',
                         "<bad>", {"sit.test": hub})
        check(False, "hub_loop rejects a placement")
    except RecipeError as e:
        check("placement is fixed" in str(e), f"hub_loop rejects a placement: {str(e)[:80]}")


def hub_pose_smoke(check, bands) -> None:
    """CPU smoke test of `kimodo_backend.build_constraints` with start + end poses from a written SOMA-77 BVH."""
    try:
        import torch
        from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
        from kimodo.geometry import axis_angle_to_matrix
        from kimodo.motion_rep.conditioning import build_condition_dicts
        from kimodo.motion_rep.feature_utils import compute_heading_angle
        from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
        from kimodo.skeleton import SOMASkeleton30
    except ImportError as e:
        print(f"skip  hub-pose constraint smoke test (torch / Kimodo not importable: {e})")
        return
    from .kimodo_backend import HubPoseConstraint, build_constraints

    sk30 = SOMASkeleton30()
    sk77 = sk30.somaskel77
    g = torch.Generator().manual_seed(7)
    T, k = 30, 12
    aa = (torch.rand(T, 77, 3, generator=g) - 0.5) * 0.6  # every joint bent up to ~0.3 rad
    aa[:, 0] = torch.tensor([0.1, math.radians(70.0), -0.05])  # pelvis turned ~70 deg (and tilted) in the take
    local = axis_angle_to_matrix(aa)
    pelvis = torch.tensor([[0.4, 0.62, -1.2]]).repeat(T, 1)
    with tempfile.TemporaryDirectory() as d:
        for rest in ("tpose", "seed"):
            src = Path(d) / f"take_{rest}.bvh"
            save_motion_bvh(str(src), local, pelvis, skeleton=sk77, fps=30, standard_tpose=rest == "tpose")
        hubs = parse_hubs_text(HUBS_TEST.replace("x/stand.bvh", (Path(d) / "take_tpose.bvh").as_posix())
                               .replace("x/sit.bvh", "take_seed.bvh").replace("frame = 1", f"frame = {k}").replace("frame = 2", f"frame = {k}"), "<hubs>")
        recipe = POSES.replace('root = "path:0,0.5"', 'root = "path:0.3,0.5"')
        job = next(j for j in plan(parse_recipe_text(recipe, "<poses>", hubs), bands) if j.phase == "enter")
        cons = build_constraints(job.constraints, sk30, "cpu", hub_root=Path(d))
        poses = [c for c in cons if isinstance(c, HubPoseConstraint)]
        ok_frames = [c.inner.frame_indices.tolist() for c in poses] == [[0], [job.num_frames - 1]]
        check(len(poses) == 2 and ok_frames and [c.rest for c in poses] == ["tpose", "seed"],
              f"build_constraints: full-body keys at frames 0 and {job.num_frames - 1}, T-pose and BONES-SEED rest detected")

        # Reference: Kimodo's own BVH reader (77 joints), frame k, moved by hand to the placement the recipe asked for.
        motion, _ = bvh_to_kimodo_motion(Path(d) / "take_tpose.bvh", sk77, standard_tpose=True)
        ref = motion["posed_joints"][k].float()
        p0 = ref[sk77.root_idx].clone()
        h0 = float(compute_heading_angle(ref[None, None], sk77)[0, 0])
        body = [n for n in sk30.bone_order_names if not n.endswith(("ThumbEnd", "MiddleEnd"))]
        i30 = [sk30.bone_order_names.index(n) for n in body]
        i77 = [sk77.bone_order_names.index(n) for n in body]
        errs = []
        for c, (x, z), head in ((poses[0], (0.0, 0.0), 0.0), (poses[1], (0.3, 0.5), 0.0)):
            th = head - h0
            rot = torch.tensor([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]])
            want = (ref - p0) @ rot.T + torch.tensor([x, float(p0[1]), z])
            got = c.inner.global_joints_positions[0]
            errs.append(float((got[i30] - want[i77]).abs().max()))
        hd = [float(compute_heading_angle(c.inner.global_joints_positions[None], sk30)[0, 0]) for c in poses]
        check(max(errs) < 1e-4 and max(abs(h) for h in hd) < 1e-4,
              f"hub joints = the BVH frame after the canonical transform ({len(body)} body joints, max {max(errs):.2e} m; "
              f"seed-rest copy agrees), heading +Z, pelvis {float(p0[1]):.2f} m kept")
        idx, data = build_condition_dicts(cons)
        n30 = sk30.nbjoints
        check(len(torch.cat(idx["global_joints_positions"])) == 2 * n30 and len(torch.cat(idx["smooth_root_2d"])) >= 2
              and float(torch.cat(data["root_y_pos"])[-1]) == float(pelvis[0, 1]),
              f"conditioning dicts: {2 * n30} joint-position keys, smooth root XZ keyed at the pose frames, root height kept")
        _, mask = KimodoMotionRep(sk30, KIMODO_FPS).create_conditions_from_constraints(cons, job.num_frames, False, "cpu")
        keyed = mask.any(-1).nonzero().flatten().tolist()
        check(keyed == [0, job.num_frames - 1], f"Kimodo's create_conditions accepts them (no stats needed): frames {keyed} conditioned")


def main() -> int:
    fails = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal fails
        print(("ok    " if cond else "FAIL  ") + msg)
        fails += 0 if cond else 1

    bands = load_bands(HERE / "default_bands.json")
    seat_families = {"seat_floor", "seat_chair", "seat_bench", "seat_stool", "seat_chair_old"}  # AN26a sit-batch1
    check(bands.placeholder and set(bands.points) == {"seat", "table", "counter", "bed", "ground"} | seat_families,
          "default_bands.json: 5 placeholder kinds + 5 seat family kinds")
    recipes = sorted(p for p in (HERE / "recipes").glob("*.toml") if p.name != "hubs.toml" and not p.name.startswith("links-") and not p.name.endswith("-links.toml"))
    acts = load_recipes(recipes)
    jobs = plan(acts, bands)
    check(len(acts) >= 3, f"{len(recipes)} recipe files, {len(acts)} activities parse")
    names = [j.outputs(Path("out"), n)[0].as_posix() for j in jobs for n in j.variants]
    check(len(names) == len(set(names)), f"{len(jobs)} jobs, {len(names)} unique output paths")
    sit = next(j for j in jobs if j.activity == "sit.chair.idle")
    check(sit.constraints["pelvis_height"]["y"][0] == 0.55 and sit.band == "seat45", "sit.chair.idle: band.seat+0.10 -> 0.55 m, band seat45")
    check(names[0].startswith("out/") and names[0].count(".") >= 4, f"path layout {names[0]}")
    lean = [j for j in jobs if j.activity == "lean.wall.arms_crossed"]
    check({j.phase for j in lean} == {"enter", "loop"} and all(j.band == "any" for j in lean), "lean: enter + loop, kind without bands -> band 'any'")

    good = parse_recipe_text(GOOD, "<good>")
    gj = plan(good, bands, max_batch=2)
    check(len(gj) == 4 and [len(j.variants) for j in gj] == [2, 1, 2, 1], "variants split into GPU batches (3 at batch 2)")
    loop = next(j for j in gj if j.phase == "loop")
    hr = loop.constraints["hand_r"]
    check(hr["pos"][0] == [-0.15, 0.8, 0.4] and hr["root_xz"][0] == [0.0, 0.0],
          f"hand_r resolved to {hr['pos'][0]} (right = -X, table 0.75 + 0.05), root pinned at the origin")
    check(len(hr["frames"]) <= MAX_KEYFRAMES and hr["frames"][-1] == loop.num_frames - 1, f"keyframes capped at {MAX_KEYFRAMES}: {len(hr['frames'])}")
    enter = next(j for j in gj if j.phase == "enter")
    check(enter.constraints["pelvis_height"]["frames"] == [enter.num_frames - 1], "at = 'end' -> last frame")

    keyed = plan(parse_recipe_text(KEYED, "<keyed>"), bands)[0].constraints
    ph, rt, hk = keyed["pelvis_height"], keyed["root"], keyed["hand_r"]
    check(ph["frames"] == [0, 15, 59] and ph["y"] == [0.95, 0.5, 0.95], f"pelvis_height groups merged by frame: {ph}")
    check(rt["frames"] == [0, 30, 59] and rt["points"] == [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], f"timed path (origin inserted at 0 s): {rt}")
    check(hk["frames"] == [9, 24] and hk["pos"] == [[-0.3, 1.7, 0.1], [0.3, 0.8, 0.5]] and hk["root_xz"] == [[0.0, 0.3], [0.0, 0.8]],
          f"hand_r groups + root XZ on the path: {hk}")

    profile_checks(check, bands)
    loco = plan(load_recipes([HERE / "recipes" / "loco-batch1.toml"]), bands)
    rows = "\n".join(plan_line(j) for j in loco)
    check(hashlib.sha256(rows.encode()).hexdigest() == LOCO_BATCH1_PLAN_SHA and all(j.cfg_weight is None for j in loco),
          f"loco-batch1.toml plans byte-identically to before AN4a ({len(loco)} rows, no cfg_weight)")
    loco2 = plan(load_recipes([HERE / "recipes" / "loco-batch2.toml"]), bands)
    rows2 = "\n".join(plan_line(j) for j in loco2)
    check(hashlib.sha256(rows2.encode()).hexdigest() == LOCO_BATCH2_PLAN_SHA,
          f"loco-batch2.toml plans byte-identically to before AN44a ({len(loco2)} rows, no hubs)")
    hub_checks(check, bands)

    for label, (body, needle) in BAD.items():
        try:
            parse_recipe_text("[[activity]]\n" + body + "\n", "<bad>")
            check(False, f"rejects {label}")
        except RecipeError as e:
            check(needle in str(e), f"rejects {label}: {e}")
    try:
        plan(parse_recipe_text('[[activity]]\nid = "a.b"\npoint = "seat"\nphases = ["loop"]\nduration = 3.0\nprompt = "x"\nconstraints.loop = { pelvis_height = "band.stool" }\n', "<bad>"), bands)
        check(False, "rejects band kind missing from bands.json")
    except RecipeError as e:
        check("no band kind 'stool'" in str(e), f"rejects band kind missing from bands.json: {e}")

    with tempfile.TemporaryDirectory() as d:
        j = jobs[0]
        for n in j.variants:
            for p in j.outputs(Path(d), n):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
        check(j.done(Path(d)) and not jobs[1].done(Path(d)), "resume: a job with all outputs present is done")

    grade_checks(check)
    hubcheck_checks(check)
    link_checks(check)

    print(f"selftest: {'FAILED ' + str(fails) if fails else 'all ok'}")
    return 1 if fails else 0
