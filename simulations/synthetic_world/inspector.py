"""Episode inspector — human readout, not a trainer."""

from __future__ import annotations

from typing import Any


def _xy(row: list) -> str:
    if len(row) < 2:
        return "?"
    return f"({float(row[0]):.2f}, {float(row[1]):.2f})"


def _motion(cur: list, nxt: list) -> str:
    if len(cur) < 2 or len(nxt) < 2:
        return "unknown"
    dx = float(nxt[0]) - float(cur[0])
    dy = float(nxt[1]) - float(cur[1])
    if abs(dx) < 0.08 and abs(dy) < 0.08:
        return "stopped"
    if abs(dx) >= abs(dy):
        return "moves left" if dx < 0 else "moves right"
    return "moves down" if dy < 0 else "moves up"


def _iter_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, list):
            rows.extend(x for x in item if isinstance(x, dict))
    return rows


def format_episode(record: dict[str, Any], *, max_frames: int = 8) -> str:
    lines = [
        f"episode {record.get('episode_id', '')}",
        f"seed {record.get('seed')}  scenario {record.get('scenario')}",
        "",
    ]
    worlds = record.get("world_states", [])
    actions = record.get("actions", [])
    futures = record.get("future_branches", [])
    events = record.get("events", [])
    outcome_rows = _iter_rows(record.get("outcomes", [])) + _iter_rows(record.get("corrections", []))
    n = min(len(worlds), max_frames)
    for t in range(n):
        world = worlds[t]
        action = actions[t] if t < len(actions) else {}
        lines.append(f"TIME {t}")
        lines.append("Entities:")
        ids = world.get("entity_id", [])
        kinds = world.get("kind", [])
        xy = world.get("xy", [])
        pos_by_id = {int(eid): xy[i] for i, eid in enumerate(ids) if i < len(xy)}
        for i, eid in enumerate(ids):
            kind = kinds[i] if i < len(kinds) else "?"
            pos = _xy(xy[i]) if i < len(xy) else "?"
            lines.append(f"{eid} {kind} {pos}")
        lines.append("Belief:")
        belief = record.get("belief_states") or []
        shown = False
        if t < len(belief) and isinstance(belief[t], dict):
            for ent in belief[t].get("entities", []):
                lines.append(f"{ent.get('id')} confidence {float(ent.get('confidence', 0.0)):.1f}")
                shown = True
        if not shown:
            vis = []
            if t < len(record.get("observations", [])):
                vis = record["observations"][t].get("visible_ids", [])
            lines.append(f"visible {vis}")
        obj = action.get("objective", "NONE")
        tgt = action.get("target", [])
        lines.append("Action:")
        lines.append(str(obj) if not tgt else f"{obj} {_xy(tgt)}")
        lines.append("Predicted:")
        fut = futures[t] if t < len(futures) else {}
        if fut:
            for eid, path in fut.items():
                if not path:
                    continue
                cur = pos_by_id.get(int(eid), path[0])
                lines.append(f"{eid} {_motion(cur, path[0])}")
        else:
            lines.append("(no future branch)")
        lines.append("REALITY:")
        if t + 1 < len(worlds):
            nxt = worlds[t + 1]
            nxt_xy = nxt.get("xy", [])
            for i, eid in enumerate(nxt.get("entity_id", [])):
                nxy = nxt_xy[i] if i < len(nxt_xy) else []
                cur = pos_by_id.get(int(eid), nxy)
                lines.append(f"{eid} {_motion(cur, nxy)} {_xy(nxy) if nxy else '?'}")
        else:
            lines.append("(terminal frame)")
        lines.append("Correction:")
        note = "none"
        for item in outcome_rows:
            if int(item.get("frame", -1)) == t and item.get("type") == "correction":
                note = str(item.get("lesson", "velocity hypothesis revised"))
                break
        if note == "none":
            for ev in events:
                if int(ev.get("frame", -1)) == t and ev.get("type") in {"conflict", "occlusion", "disappearance"}:
                    note = str(ev["type"])
                    break
        lines.append(note)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
