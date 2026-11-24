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
    fail_file = os.environ.get("TOY_FAIL_FILE")  # tests: fail when the checkout's path ends with the file's text
    if fail_file and os.path.exists(fail_file) and str(checkout).endswith(Path(fail_file).read_text().strip()):
        sys.exit(7)
    rules = module.load_rules(Path(checkout) / "rules.txt")
    got = module.classify(task["text"], rules)
    correct = got == task["label"]
    summary = f"{task['text']!r}: got {got}, expected {task['label']}"
    factor_file = os.environ.get("TOY_FACTOR_FILE")  # tests: stands for machines that got faster or slower
    factor = 1.0
    if factor_file and os.path.exists(factor_file):  # "0.5", or "0.5 /r0-s1" for checkouts ending with that only
        value, _, where = Path(factor_file).read_text().strip().partition(" ")
        if not where or str(checkout).endswith(where):
            factor = float(value)
    result = {"score": (1.0 if correct else 0.0) * factor, "summary": summary}
    if task.get("metrics"):
        result["metrics"] = {"rules": len(rules), "labels": len({label for _, label in rules})}
    Path(out, "result.json").write_text(json.dumps(result))
    log = os.environ.get("TOY_CALL_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(f"{out}\n")  # one line per rollout that really ran


if __name__ == "__main__":
    main(*sys.argv[1:4])
