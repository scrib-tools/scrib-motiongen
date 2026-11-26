# SPDX-License-Identifier: GPL-3.0-or-later
"""Hub pose table (`recipes/hubs.toml`, AN43). Pure Python (no torch): the BVH frame is read by `kimodo_backend`.

A hub is a body hub plus a hand state, sourced from one frame of a kept SOMA-77 take. Format and rule:
motion-generation.md "Interaction pieces and hub poses".
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HUBS = HERE / "recipes" / "hubs.toml"

HUB_KEYS = {"id", "family", "body", "hands", "source", "frame", "notes"}
BODY_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*(@[a-z0-9_]+)?$")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EMPTY = "empty"


class HubError(ValueError):
    """The hubs file is invalid; the message names the file and the hub."""


@dataclass(frozen=True)
class Hub:
    id: str
    family: str
    body: str
    hands: dict = field(hash=False)  # {"L": "empty" | prop class, "R": ...}
    source: str  # BVH path as written in hubs.toml (relative to the output root, or absolute)
    frame: int
    file: str  # the hubs.toml it came from

    def source_path(self, root: Path | None) -> Path:
        """The BVH file: `source` if absolute, else relative to `root` (the run's --out; cwd when None)."""
        p = Path(self.source)
        return p if p.is_absolute() else (Path(root) if root is not None else Path.cwd()) / p


def hub_name(body: str, hands: dict) -> str:
    """Canonical hub id: the body, then `+L:<prop>` / `+R:<prop>` for each hand that holds something (L first)."""
    return body + "".join(f"+{s}:{hands[s]}" for s in ("L", "R") if hands.get(s, EMPTY) != EMPTY)


def parse_hubs_text(text: str, src: str) -> dict[str, Hub]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise HubError(f"{src}: TOML error: {e}") from e
    if set(data) - {"hub"}:
        raise HubError(f"{src}: unknown top-level keys {sorted(set(data) - {'hub'})} (expected [[hub]] tables)")
    rows = data.get("hub", [])
    if not isinstance(rows, list) or not rows:
        raise HubError(f"{src}: no [[hub]] tables")
    out: dict[str, Hub] = {}
    for i, h in enumerate(rows):
        where = f"{src}: hub[{i}]"
        if set(h) - HUB_KEYS:
            raise HubError(f"{where}: unknown keys {sorted(set(h) - HUB_KEYS)} (allowed: {sorted(HUB_KEYS)})")
        for req in ("id", "family", "body", "source", "frame"):
            if req not in h:
                raise HubError(f"{where}: missing '{req}'")
        hid = h["id"]
        where = f"{src}: hub '{hid}'"
        body, family = h["body"], h["family"]
        if not isinstance(body, str) or not BODY_RE.match(body):
            raise HubError(f"{where}: body {body!r} must be a dotted lowercase body hub (stand.idle, sit.chair, stand.idle@old)")
        if not isinstance(family, str) or not NAME_RE.match(family):
            raise HubError(f"{where}: family {family!r} must be a lowercase name ('any' = every family)")
        hands = h.get("hands", {})
        if not isinstance(hands, dict) or set(hands) - {"L", "R"}:
            raise HubError(f"{where}: hands must be a table {{ L = ..., R = ... }}")
        hands = {s: hands.get(s, EMPTY) for s in ("L", "R")}
        for s, v in hands.items():
            if not isinstance(v, str) or not NAME_RE.match(v):
                raise HubError(f"{where}: hands.{s} = {v!r}: expected 'empty' or a lowercase prop class (cup, book ...)")
        want = hub_name(body, hands)
        if hid != want:
            raise HubError(f"{where}: id must be body + hand state, here {want!r}")
        if hid in out:
            raise HubError(f"{where}: duplicate hub id")
        source, frame = h["source"], h["frame"]
        if not isinstance(source, str) or not source.lower().endswith(".bvh"):
            raise HubError(f"{where}: source {source!r} must be a .bvh path (relative to --out, or absolute)")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise HubError(f"{where}: frame must be a 0-based frame index")
        out[hid] = Hub(hid, family, body, hands, source, frame, src)
    return out


def load_hubs(path: Path = DEFAULT_HUBS) -> dict[str, Hub]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise HubError(f"{path}: cannot read hubs: {e}") from e
    return parse_hubs_text(text, str(path))


_default: dict[str, Hub] | None = None


def default_hubs() -> dict[str, Hub]:
    """`recipes/hubs.toml`, loaded once (recipes that name no hub never load it)."""
    global _default
    if _default is None:
        _default = load_hubs(DEFAULT_HUBS)
    return _default
