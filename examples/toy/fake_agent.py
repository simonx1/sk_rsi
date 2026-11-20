#!/usr/bin/env python3
"""A scripted stand-in for a coding agent, for tests and demos.

Usage: fake_agent.py --out <out_dir>
Reads the script named by FAKE_AGENT_SCRIPT and acts on RSI_ROLE, RSI_ROUND and
RSI_SAMPLE (proposer) or RSI_STAGE (selector). Exits 3 when the script has no
entry for this call, which the framework sees as a failed agent.
"""

import argparse
import json
import os
import sys
from pathlib import Path

NO_ENTRY = 3


def propose(plan, out):
    for edit in plan.get("edits", []):
        path = Path(edit["path"])  # relative to the worktree, which is the working directory
        if "append" in edit:
            with path.open("a") as handle:
                handle.write(edit["append"])
        elif "write" in edit:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit["write"])
        elif edit.get("delete"):
            path.unlink()
    if "raw_manifest" in plan:
        (out / "manifest.json").write_text(plan["raw_manifest"])
    elif "manifest" in plan:
        (out / "manifest.json").write_text(json.dumps(plan["manifest"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    out = Path(parser.parse_args().out)
    script = json.loads(Path(os.environ["FAKE_AGENT_SCRIPT"]).read_text())
    round_no = os.environ["RSI_ROUND"]
    if os.environ["RSI_ROLE"] == "proposer":
        plan = script.get("proposer", {}).get(round_no, {}).get(os.environ["RSI_SAMPLE"])
        if plan is None:
            return NO_ENTRY
        propose(plan, out)
        return 0
    stage = os.environ["RSI_STAGE"]
    if round_no not in script.get(stage, {}):
        return NO_ENTRY
    answer = script[stage][round_no]
    if stage == "screen":
        rows = [{"sample": s, "keep": keep, "reason": "scripted"} for s, keep in answer.items()]
        (out / "screen.json").write_text(json.dumps({"screen": rows}))
    else:
        (out / "decision.json").write_text(json.dumps({"accept": answer, "reason": "scripted"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
