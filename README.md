# sk_rsi

Put any project into a recursive self-improvement (RSI) loop, with a memory of what was tried.

It implements the method of *Self-Improvement Memory Enables Continual Recursive Self-Improvement*
(Yu et al., arXiv:2611.18856) as a small command-line tool. The project being improved supplies one
config file and one command that scores a task. Everything else is here.

## How it works

Each **round**:

1. The current version of the project is scored on the **evolve** tasks.
2. Several **proposer** agents each make one edit in their own git worktree and predict which failing
   tasks the edit will fix.
3. Edits that touch a path they may not touch, fail the project's check, or have an invalid manifest
   are dropped. A **selector** agent may drop more.
4. The surviving edits are scored on the evolve tasks.
5. The selector keeps one edit or none. An edit that scores more than a set tolerance below the
   current version cannot be kept.
6. The kept edit is recorded in the **Self-Improvement Memory**, `.rsi/sim.json`: its root cause, what
   it changed and why, which tasks it was predicted to fix, and which it really fixed, missed or
   broke.

A second set of tasks, **held-out**, is scored for people only. The agents never see those tasks,
their answers or their results.

The memory is given in full to the proposer and the selector every round. When the model or the task
set changes, the memory carries over, so the agents know why each inherited edit exists and in which
setting it was measured.

A run works on its own branch and ends with a pull request. Nothing reaches your main branch any
other way.

## Install

Needs Python 3.11 or newer, `git`, and for the default agents the `claude` CLI. `gh` is needed only
to open pull requests.

```bash
python3 -m pip install -e .
bin/install-skill        # optional: the /rsi skill for Claude Code
```

## Onboard a project

In the project's repo:

```bash
rsi init                 # writes .rsi/config.toml, fully commented
```

Then edit the config. An example:

```toml
[project]
name = "ticket-triage"
base_branch = "main"

[setting]
benchmark = "replay"
model = "claude-opus-5-5"

[harness]
editable = ["prompts/**"]
frozen = ["prompts/tests/**", "bin/score-ticket"]

[tasks]
rollouts = 2
pass_threshold = 0.5
heldout = "endpoints"

[commands]
run_task = "bin/score-ticket {task_file} {checkout} {out}"
check = "bin/test && bin/lint"
ingest = "bin/outcomes --since {since}"

[budget]
rounds = 5
candidates = 3
parallel = 3
tolerance = 1.0
```

Write the tasks to `~/.rsi/<name>/private/tasks.jsonl`, one JSON object per line with a unique `id`.
The file holds the expected answers, so it must stay outside the repo.

Write the `run_task` command. It is called once per task and rollout:

| Placeholder | Value |
|---|---|
| `{task_file}` | A file holding this task's JSON object |
| `{task_id}` | The task's id |
| `{checkout}` | A git worktree of the version being scored; also the working directory |
| `{out}` | An empty directory for this rollout |
| `{rollout}` | Rollout number, from 1 |

It must write `{out}/result.json` with a `score` from 0 to 1 and an optional `summary`. Anything else
it leaves in `{out}` is shown to the proposer for evolve tasks. A non-zero exit, a timeout or an
invalid result counts as a score of 0. Placeholders arrive shell-quoted; do not quote them again.

Commit the config and the scorer, then:

```bash
rsi validate             # config, task file, split sizes, check
rsi run --dry-run        # how many task runs a run would start
rsi run                  # start, or resume after an interruption
```

## Commands

```bash
rsi init                 # write a commented config
rsi validate             # check the setup
rsi run                  # resume the unfinished run, or start one
rsi continue             # fold queued live tasks into the benchmark, then run
rsi status               # the latest run: round, step, scores, pull request
rsi ingest               # run the ingest command now
rsi promote              # push a finished run's branch and open its pull request
rsi sim show             # print the memory
rsi sim merge P2 P1      # merge one problem entry into another
```

`rsi run` and `rsi continue` take `--rounds N`, `--from BRANCH`, `--no-promote`, `--new` (abandon the
unfinished run) and `--dry-run`. Every command takes `--project PATH`.

## Live outcomes

The optional `ingest` command prints one JSON record per line:

```json
{"kind": "evidence", "id": "reopened-88121", "summary": "Customer reopened a ticket the harness routed", "task_id": "t-4812"}
{"kind": "task", "id": "live-t-5120", "summary": "Worth replaying", "task": {"id": "t-5120"}}
```

Evidence is shown to the agents as context and never decides which edit is kept. Evidence about a
held-out task is withheld. Task records are queued and join the benchmark only on `rsi continue`,
which raises the benchmark version (`replay-v1` to `replay-v2`).

## Where things live

| Place | Contents |
|---|---|
| `.rsi/config.toml` in the project | The config |
| `.rsi/sim.json`, `.rsi/log.md` on the run branch | The memory and a per-round log of evolve scores |
| `~/.rsi/<name>/private/` | Tasks, the split, run state, held-out results. Never given to agents |
| `~/.rsi/<name>/runs/<run>/` | Evolve results, each round's candidates, agent prompts and logs |
| `~/.rsi/<name>/worktrees/<run>/` | Git worktrees. Your working copy is never touched |

Set `RSI_HOME` to use a directory other than `~/.rsi`.

## Guards

- A candidate may only change `harness.editable` paths. `harness.frozen` and `.rsi/` are off limits,
  and a `run_task`, `check` or `ingest` command that the loop could rewrite is refused.
- The default agents have file tools and no shell, plus a rule denying the private directory. A custom
  `[agents]` command with shell access gives up that guarantee; `rsi validate` warns about it. This is
  a permission rule, not an operating-system sandbox.
- A run stops before optimizing when too many rollouts fail to run at all.
- Budgets cap rounds, candidates, parallel runs, timeouts and total active time.

## Tests

```bash
bin/test                          # everything, in under a minute
RSI_SMOKE=1 bin/test -k smoke     # one round on the toy project with the real claude CLI (costs tokens)
```

`examples/toy/` is a tiny project used by the tests: a rules file as the harness, a deterministic
scorer, twelve tasks and a scripted agent.
