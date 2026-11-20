"""Run tasks against a checkout and aggregate their scores."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from . import shell
from .config import Config
from .state import Layout, write_json_atomic

MARKER = "rollout.json"  # written last; a rollout directory without it is unfinished


@dataclass(frozen=True)
class Rollout:
    task_id: str
    rollout: int
    score: float
    error: str | None   # set for an infra failure: the command did not produce a valid result
    summary: str


def _read_result(out_dir: Path) -> tuple[float, str]:
    try:
        obj = json.loads((out_dir / "result.json").read_text())
    except FileNotFoundError:
        raise ValueError("result.json missing") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("result.json is not JSON") from None
    except OSError:
        raise ValueError("result.json could not be read") from None
    if not isinstance(obj, dict):
        raise ValueError("result.json is not an object")
    score = obj.get("score")
    # NaN fails the range test, and so does an integer too large to be a float.
    valid = isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1
    if not valid:
        raise ValueError("score must be a number from 0 to 1")
    summary = obj.get("summary", "")
    return float(score), summary if isinstance(summary, str) else str(summary)


def run_rollout(cfg: Config, task: dict, rollout: int, checkout: Path, out_dir: Path, benchmark: str) -> Rollout:
    """Run one task once. A finished rollout is read back instead of run again."""
    marker = out_dir / MARKER
    if marker.is_file():
        return Rollout(**json.loads(marker.read_text()))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # The task holds its expected answer. Its file goes in the private directory, which agents are
    # denied, so a copy left behind by a killed run is still out of their reach.
    task_dir = Layout(cfg.name).private / "tmp"
    task_dir.mkdir(parents=True, exist_ok=True)
    handle, task_path = tempfile.mkstemp(prefix="task-", suffix=".json", dir=task_dir)
    try:
        with os.fdopen(handle, "w") as fh:
            json.dump(task, fh)
        command = shell.render(
            cfg.run_task,
            {"task_file": task_path, "task_id": task["id"], "checkout": checkout, "out": out_dir, "rollout": rollout},
        )
        env = {
            "RSI_TASK_ID": task["id"],
            "RSI_ROLLOUT": str(rollout),
            "RSI_MODEL": cfg.model,
            "RSI_BENCHMARK": benchmark,
        }
        done = shell.run(command, cwd=checkout, timeout=cfg.task_timeout_s, env=env)
    finally:
        os.unlink(task_path)
    (out_dir / "stdout.log").write_text(done.stdout)
    (out_dir / "stderr.log").write_text(done.stderr)
    score, summary, error = 0.0, "", None
    if done.timed_out:
        error = f"timeout after {cfg.task_timeout_s}s"
    elif done.returncode != 0:
        error = f"exit code {done.returncode}"
    else:
        try:
            score, summary = _read_result(out_dir)
        except ValueError as problem:
            error = str(problem)
    result = Rollout(task["id"], rollout, score, error, summary)
    write_json_atomic(marker, asdict(result))
    return result


@dataclass
class EvalResult:
    task_ids: list[str]
    rollouts: list[Rollout]

    def task_scores(self) -> dict[str, float]:
        scores: dict[str, list[float]] = {task_id: [] for task_id in self.task_ids}
        for r in self.rollouts:
            scores[r.task_id].append(r.score)
        return {task_id: sum(v) / len(v) for task_id, v in scores.items() if v}

    def score(self) -> float:
        per_task = self.task_scores()
        if not per_task:
            return 0.0
        return round(100 * sum(per_task.values()) / len(per_task), 2)

    def passing(self, threshold: float) -> set[str]:
        return {task_id for task_id, s in self.task_scores().items() if s >= threshold}

    def passed_total(self, threshold: float) -> tuple[int, int]:
        return sum(1 for r in self.rollouts if r.score >= threshold), len(self.rollouts)

    def infra_failure_rate(self) -> float:
        if not self.rollouts:
            return 0.0
        return sum(1 for r in self.rollouts if r.error) / len(self.rollouts)

    def table(self, threshold: float) -> list[dict]:
        per_task = self.task_scores()
        rows = []
        for task_id in self.task_ids:
            mine = [r for r in self.rollouts if r.task_id == task_id]
            notes = [r.summary or r.error or "" for r in mine]
            rows.append({
                "task": task_id,
                "score": round(per_task.get(task_id, 0.0), 4),
                "passes": per_task.get(task_id, 0.0) >= threshold,
                "errors": sum(1 for r in mine if r.error),
                "summary": next((n for n in notes if n), ""),
            })
        return rows


def evaluate_many(
    cfg: Config,
    task_list: list[dict],
    targets: dict[str, tuple[Path, Path]],
    benchmark: str,
) -> dict[str, EvalResult]:
    """Score several checkouts at once, at most cfg.parallel rollouts at a time.

    `targets` maps a name to (checkout, results_dir).
    """
    collected: dict[str, list[Rollout]] = {name: [] for name in targets}
    pool = ThreadPoolExecutor(max_workers=cfg.parallel)
    try:
        futures = {}
        for name, (checkout, results_dir) in targets.items():
            for task in task_list:
                for k in range(1, cfg.rollouts + 1):
                    out_dir = results_dir / task["id"] / f"r{k}"
                    futures[pool.submit(run_rollout, cfg, task, k, checkout, out_dir, benchmark)] = name
        for future in as_completed(futures):
            collected[futures[future]].append(future.result())
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()
    order = {task["id"]: i for i, task in enumerate(task_list)}
    results = {}
    for name, rollouts in collected.items():
        rollouts.sort(key=lambda r: (order[r.task_id], r.rollout))
        results[name] = EvalResult([task["id"] for task in task_list], rollouts)
    return results


def evaluate(cfg: Config, task_list: list[dict], checkout: Path, results_dir: Path, benchmark: str) -> EvalResult:
    return evaluate_many(cfg, task_list, {"only": (checkout, results_dir)}, benchmark)["only"]
