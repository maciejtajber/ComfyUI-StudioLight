#!/usr/bin/env python3
"""
StudioLight workflow migrator.

Brings StudioLightRig `widgets_values` (graph format) / `inputs` (API format) in line with the current
node schema so old workflows cannot shift widgets (e.g. 5500 landing in key_intensity).

    python tools/migrate_workflows.py              # dry run, report only (exit 1 if anything needs work)
    python tools/migrate_workflows.py --write      # rewrite migratable files in place
    python tools/migrate_workflows.py a.json b.json

Legacy layouts handled (rig node only):
  22 values  pre-distance layout (preset .. rim_modifier)
  26 values  22 + four obsolete trailing widgets (dropped)
  26-30      current layout missing trailing optionals (defaults appended; schema validated)
Migration is BEHAVIOUR-PRESERVING for legacy layouts: distances 1.8 m (attenuation 1.0), normal_boost 1.0, ao_anchoring 0.0.

Workflows that link to rig outputs which no longer exist (old `conditioning` / `lit_composite` slots) are reported as
OBSOLETE and never rewritten: their graph needs rebuilding, not widget patching.
"""
import argparse, glob, json, os, sys, math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from nodes.studio_light import StudioLightRig  # noqa: E402

IT = StudioLightRig.INPUT_TYPES()
REQUIRED = list(IT["required"])
WIDGET_OPTIONAL = [k for k, v in IT["optional"].items() if v[0] != "IMAGE"]
WIDGETS = REQUIRED + WIDGET_OPTIONAL
OUTPUT_TYPES = list(StudioLightRig.RETURN_TYPES)
OUTPUT_NAMES = list(StudioLightRig.RETURN_NAMES)
LEGACY_BASE = 22  # preset, framing, ambient, invert_y + 3 lights x (enabled, az, el, int, kelvin, mod)


def default_of(name):
    spec = IT["required"].get(name) or IT["optional"].get(name)
    t = spec[0]
    if isinstance(t, list):
        return spec[1].get("default", t[0]) if len(spec) > 1 else t[0]
    return spec[1]["default"]


def values_valid(values, names):
    """Validate serialized values before interpreting or rewriting a layout."""
    if len(values) != len(names):
        return False
    for value, name in zip(values, names):
        spec = IT["required"].get(name) or IT["optional"][name]
        kind = spec[0]
        if isinstance(kind, list):
            if value not in kind:
                return False
        elif kind == "BOOLEAN":
            if not isinstance(value, bool):
                return False
        elif kind in ("FLOAT", "INT"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
            if kind == "INT" and not isinstance(value, int):
                return False
            if not spec[1].get("min", -math.inf) <= value <= spec[1].get("max", math.inf):
                return False
    return True


def migrate_values(wv):
    """Returns validated values or refuses an unknown/invalid layout."""
    n = len(wv)
    # Current layouts are recognized by schema, not count alone: 26 values
    # can mean today's required widgets OR an old layout with framing.
    if len(REQUIRED) <= n <= len(WIDGETS) and values_valid(wv, WIDGETS[:n]):
        out = list(wv) + [default_of(k) for k in WIDGETS[n:]]
        return out, "current" if n == len(WIDGETS) else f"short-{n}"
    if n in (22, 26) and isinstance(wv[1], str) and wv[1]:
        base = list(wv[:LEGACY_BASE])
        out = [base[0], base[2], base[3]]
        for i in range(3):
            en, az, el, inten, kel, mod = base[4 + 6 * i: 10 + 6 * i]
            out += [en, az, el, 1.8, inten, kel, mod]
        out += [1.0, 0.0]
        out += [default_of(k) for k in WIDGET_OPTIONAL]
        if values_valid(out, WIDGETS):
            return out, f"legacy-{n}"
    return None, f"unrecognised or invalid widget layout ({n} values)"


def outputs_ok(links, rig_ids):
    bad = []
    for l in links:  # [id, origin, origin_slot, target, target_slot, type]
        if l[1] in rig_ids:
            if not isinstance(l[2], int) or isinstance(l[2], bool) or not 0 <= l[2] < len(OUTPUT_TYPES) or l[5] != OUTPUT_TYPES[l[2]]:
                bad.append(l)
    return bad


def fix_graph(wf):
    rigs = [n for n in wf.get("nodes", []) if n.get("type") == "StudioLightRig"]
    if not rigs:
        return "skip", "no StudioLightRig", False
    bad = outputs_ok(wf.get("links", []), {n["id"] for n in rigs})
    if bad:
        return "obsolete", f"{len(bad)} link(s) use removed rig outputs (e.g. slot {bad[0][2]} as {bad[0][5]})", False
    changed = False
    notes = []
    cur_inputs = set(IT["required"]) | set(IT["optional"])
    for n in rigs:
        stale_in = [i["name"] for i in n.get("inputs", []) if i["name"] not in cur_inputs]
        if stale_in:  # removing inputs would shift link target slots; rebuild the graph instead
            return "obsolete", f"rig node carries removed inputs {stale_in[:4]}", False
        for slot, o in enumerate(n.get("outputs", [])):
            if slot >= len(OUTPUT_NAMES) or o.get("name") != OUTPUT_NAMES[slot]:
                return "obsolete", f"rig output slot {slot} is '{o.get('name')}', schema expects '{OUTPUT_NAMES[min(slot, len(OUTPUT_NAMES)-1)]}'", False
    for n in rigs:
        new, note = migrate_values(n.get("widgets_values", []))
        if new is None:
            return "obsolete", note, False
        notes.append(note)
        if new != n.get("widgets_values"):
            n["widgets_values"] = new
            changed = True
        outs = n.setdefault("outputs", [])
        for slot in range(len(outs), len(OUTPUT_TYPES)):
            outs.append({"name": OUTPUT_NAMES[slot], "type": OUTPUT_TYPES[slot], "links": None, "slot_index": slot})
            changed = True
    return ("migrated" if changed else "ok"), ",".join(notes), changed


def fix_api(d):
    rigs = {k: v for k, v in d.items() if isinstance(v, dict) and v.get("class_type") == "StudioLightRig"}
    if not rigs:
        return "skip", "no StudioLightRig", False
    changed = False
    for rid, node in rigs.items():
        for k, v in d.items():
            for iname, val in (v.get("inputs", {}) if isinstance(v, dict) else {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) == rid and (not isinstance(val[1], int) or isinstance(val[1], bool) or not 0 <= val[1] < len(OUTPUT_TYPES)):
                    return "obsolete", f"node {k}.{iname} reads removed rig output slot {val[1]}", False
        unknown = [i for i in node["inputs"] if i not in IT["required"] and i not in IT["optional"]]
        if unknown:
            return "obsolete", f"rig has removed inputs {unknown[:4]}", False
        for name in REQUIRED:
            if name not in node["inputs"]:
                legacy = {"key_distance": 1.8, "fill_distance": 1.8, "rim_distance": 1.8, "normal_boost": 1.0, "ao_anchoring": 0.0}
                node["inputs"][name] = legacy.get(name, default_of(name))
                changed = True
    return ("migrated" if changed else "ok"), "api", changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    paths = a.paths or sorted(glob.glob(os.path.join(ROOT, "workflows", "*.json")))
    worst = 0
    obsolete = 0
    for p in paths:
        data = json.load(open(p))
        status, note, changed = (fix_graph if "nodes" in data else fix_api)(data) if isinstance(data, dict) else ("skip", "?", False)
        if status == "skip":
            continue
        print(f"{status.upper():9s} {os.path.relpath(p, ROOT)}  [{note}]")
        if status == "obsolete":
            worst = 1
            obsolete += 1
        elif changed:
            worst = max(worst, 1)
            if a.write:
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)
    return 1 if (worst and not a.write) or obsolete else 0


if __name__ == "__main__":
    sys.exit(main())
