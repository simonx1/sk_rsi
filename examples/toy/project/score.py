#!/usr/bin/env python3
"""run_task for the toy project. Usage: score.py <task_file> <checkout> <out>"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # a rollout must not leave files in the checkout


def main(task_file, checkout, out):
    task = json.loads(Path(task_file).read_text())
    time.sleep(float(os.environ.get("TOY_DELAY", "0")))  # lets a test catch the loop mid-run
    spec = importlib.util.spec_from_file_location("classify", Path(checkout) / "classify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    got = module.classify(task["text"], module.load_rules(Path(checkout) / "rules.txt"))
    correct = got == task["label"]
    summary = f"{task['text']!r}: got {got}, expected {task['label']}"
    Path(out, "result.json").write_text(json.dumps({"score": 1.0 if correct else 0.0, "summary": summary}))
    log = os.environ.get("TOY_CALL_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(f"{out}\n")  # one line per rollout that really ran


if __name__ == "__main__":
    main(*sys.argv[1:4])
