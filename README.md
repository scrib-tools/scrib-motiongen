<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# scrib-motiongen: offline clip factory (Kimodo batch worker)

Part of [Scrib tools](https://github.com/scrib-tools). Turns activity recipes (`motiongen/recipes/*.toml`) into
SOMA-77 BVH clips with NVIDIA Kimodo, in one persistent process. Offline tool only: never part of a game or its bake.
GPL-3.0-or-later (`LICENSE`). Moved out of the [Scrib](https://github.com/scrib-tools/Scrib) repo on 2026-11-23 with its history.

Spec and the game-side contract (import onto a rig, census `bands.json`, clip preview, `.prov.json` sidecar):
Scrib's [motion-generation.md](https://github.com/scrib-tools/Scrib/blob/main/bevyrrowind/docs/specs/motion-generation.md) (MG1-MG10).

## Setup (once)

- Kimodo clone + venv outside the repo (Apache-2.0, imported as a library, never edited):
  `git clone https://github.com/nv-tlabs/kimodo C:\mbtest\kimodo`, then `uv venv` + `uv pip install -e .` in it
  (torch with CUDA, e.g. cu128). Python: `C:\mbtest\kimodo\.venv\Scripts\python.exe` (3.11).
- Hugging Face: `huggingface-cli login`, and accept the Meta Llama 3 licence for
  `meta-llama/Meta-Llama-3-8B-Instruct` (the LLM2Vec text encoder's base, ~15 GB). Weights land in the HF cache
  (`%USERPROFILE%\.cache\huggingface\hub`): `nvidia/Kimodo-SOMA-RP-v1.2` (1.1 GB) and the two McGill-NLP LLM2Vec repos.
- RAM: the encoder needs ~16 GB (bf16 on the CPU) while it encodes new prompts; it is freed before generation.

## Commands

Install it into the Kimodo venv once (`uv pip install -e <this repo>` with that venv active), or run from the repo
root (the folder holding `motiongen/`). Paths below are from the repo root; `--out` is any folder, e.g. Scrib's
`animation packs/generated`:

```sh
PY=/c/mbtest/kimodo/.venv/Scripts/python.exe
$PY -m motiongen selftest                                    # parse recipes/, plan, format checks (no GPU)
$PY -m motiongen plan motiongen/recipes/seed.toml --out "<scrib>/animation packs/generated"   # dry run
$PY -m motiongen run  motiongen/recipes/seed.toml --out "<scrib>/animation packs/generated" [--only 'sit.*'] [--steps 50]
$PY -m motiongen bench motiongen/recipes/seed.toml --out <dir>   # speed matrix: steps, batch, precision, postprocess
$PY -m motiongen joints                                      # rewrite soma77_joints.txt
```

`grade <dir> [--out grades.json] [--verdicts [verdicts.toml]]` (AG1, numpy only): physical reject gates per clip on the
source BVH (`skate` >= 60 cm/s, `jerk` p99 >= 5000 m/s^3, turn-in-place `turn_mag` > 9 deg off the prompt and `no_step`;
a wrong-way turn is only `mirror_candidate`) and prompt-group triage (>= 2/3 of a group's takes gated = failing).
Writes `<dir>/grades.json` atomically (stem -> type, gates, mirror_candidate, metrics; `_groups`), prints a table per group;
`--verdicts` prints gated vs the owner's reject / maybe / keep. Design: Scrib's [17-autograde](https://github.com/scrib-tools/Scrib/blob/main/superfargoth/docs/research/animation/17-autograde-2026-11-12.md).

`hubcheck <dir> [--hub-root <dir>] [--out hubcheck.json]` (AN44b, numpy only): for takes whose `.prov.json` names
`from_hub` / `to_hub`, the first / last frame vs that hub pose (`recipes/hubs.toml`; relative sources resolve against
`--hub-root`, default `<dir>`'s parent): local rotation error over the 26 body joints (max / mean deg; take and hub must
share a rest pose), hand / foot / pelvis error in the character frame (cm; pelvis XZ at 0, heading aligned, height kept)
and pelvis height error, plus foot slide (in-contact foot speed mean / p95 cm/s). Prints a row per take and per prompt
group; writes `<dir>/hubcheck.json` atomically. Takes without hub keys are skipped.

`run` options: `--bands <bands.json>` (default `<out>/bands.json` from the MG4 census if present, else
`default_bands.json`, placeholders), `--batch N` (max samples per GPU batch, default 8), `--steps N` (default 50),
`--precision fp32|fp16|bf16`, `--no-postprocess`, `--bvh-rest tpose|seed`, `--encoder-device cpu|cuda`,
`--cache <dir>`, `--hub-root <dir>` (where relative hub sources resolve, default `--out`), `--force`. Resumable: a job whose `.bvh` + `.prov.json` all exist is skipped.

Output: `<out>/<activity>/<phase>/<activity>.<phase>.<band>.<nn>.bvh` + `.prov.json` (model, revision, prompt, seed,
constraints, steps, guidance, metrics, date, tool version) and `<out>/jobs.jsonl` (one line per GPU batch: stage timings,
VRAM, per-sample metrics). Prompt embeddings: `<out>/_cache/text-embeddings/<sha256>.npy`.

## Environment

| Variable | Use |
|---|---|
| `PYTHONIOENCODING=utf-8` | Windows console output |
| `TEXT_ENCODER_MODE=local` | Kimodo: no remote text encoder service |
| `TEXT_ENCODER_DEVICE` | set by motiongen from `--encoder-device` (default `cpu`) |
| `LOCAL_CACHE=true` | set by motiongen: weights from the HF cache, no network check |
| `HF_HUB_DISABLE_SYMLINKS_WARNING=1` | quiet HF cache warning on Windows |
| `MOTIONGEN_CACHE` | embedding cache dir (overrides `<out>/_cache/text-embeddings`) |

## Recipes

Format and constraint vocabulary: spec, "Recipes (MG1)". Implemented: `pelvis_height` (meters or
`band.<kind>[+offset]`, optional `{ height, at = "all"|"start"|"end"|[seconds] }`), `root = "fixed"` (XZ + facing),
`"fixed_xz"`, `"path:x,z;x,z"` (sparse keys; `x,z@seconds` to time them), `hand_l`/`hand_r = { height, forward = 0.35,
side = 0.2, at }` (wrist position; pins the root XZ at its keyframes), `keys` (keyframe spacing, s), `start_pose` /
`end_pose` (hub poses, below). Unknown keys are errors.

Dense root path from a speed profile (one key per output frame, NVIDIA's own root-path pattern; dense root paths are
exempt from Kimodo's < 20 keyframe limit): `root = "profile:hold=0.5,ramp=1.5,speed=1.6,decel=0.9,rest=0.8,turn=90"`.
Segments run in that order, in seconds, each defaulting to 0: `hold` (still), `ramp` (smoothstep from 0 to `speed`),
steady (not a field: fills whatever the phase duration leaves), `decel` (smoothstep from `speed` to 0), `rest` (still).
`speed` (m/s, required) is along +Z; distance is the exact integral of the speed. `turn` (degrees, + = left, default 0,
needs `ramp`) swings the travel direction over the ramp and keys the root heading at the last frame. Examples: cycle
`"profile:speed=1.4"`, start `"profile:hold=0.5,ramp=1.5,speed=1.6"`, stop `"profile:speed=1.6,decel=0.9,rest=0.8"`.
Segments longer than the phase are an error.

Per activity, `cfg_weight = [text, constraint]` sets Kimodo's classifier-free guidance weights (separated CFG,
`Kimodo.__call__` `cfg_weight`); absent = Kimodo's default (2.0, 2.0). `plan` shows it as `cfg [t,c]` only when set;
the `.prov.json` records `cfg_type`, `cfg_weight` and `cfg_weight_source` (`recipe` or `kimodo default`).
Canonical space: meters, Y up, the actor starts at XZ (0, 0) facing +Z, left hand on +X.

### Hub poses (AN43)

Interaction pieces start and end on listed hub poses (Scrib spec, "Interaction pieces and hub poses"). The hubs live in
`motiongen/recipes/hubs.toml` (`motiongen/hubs.py`): `[[hub]] id, family, body, hands = { L = "empty", R = "cup" },
source = "<take>.bvh", frame = N`; the id is the body plus the hand state (`sit.chair+R:cup`), the source one frame of
a kept SOMA-77 take (relative to `--hub-root`, default `--out`, or absolute; T-pose or BONES-SEED rest, detected).
A phase names them with `start_pose = "hub:<id>"` / `end_pose = "hub:<id>"` and optional `pose_frames = N` (hold the
pose N frames, default 1). Each becomes a Kimodo `FullBodyConstraintSet` (every model joint's global position, the root
XZ, height and heading) on the phase's first / last N frames: the hub frame is cut to the model's SOMA-30 joints
(`from_SOMASkeleton77`) and posed by its FK, its pelvis put at the hub's own height and turned about the vertical.
Placement: `start_pose` at the canonical start (XZ origin, facing +Z); `end_pose` at the root path's end when the phase
has a `root` path or profile, else at the origin (Kimodo needs the root XZ keyed under joint positions, so XZ is never
free), facing +Z or the profile's final `turn` heading. A pose frame also keyed by `pelvis_height` or a hand is an
error; poses and hand keys share Kimodo's < 20 joint-position keyframes. `plan` prints `hubs a->b` (`none` for an unset
end); `.prov.json` gets `from_hub` / `to_hub`, and its `constraints` hold each pose's hub, source, source frame and
frames.
