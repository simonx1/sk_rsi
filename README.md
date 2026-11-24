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

Edits that were not kept are not in the memory, but they are not forgotten. The proposer and the
selector, when it screens, are given a list of them from this run's earlier rounds and from earlier
runs of the project, newest first, each with its outcome: why it was dropped, or what it scored and
what the tasks reported. A repeat then needs new evidence. The list is read from the history folders
and holds at most 30 edits. Agents can open this run's history folder for the details; for an earlier
run the line is all they get.

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

## Measured goals

Some goals are a measurement, not a set of right answers: how fast a service answers, what a job
costs. The measurement is noisy, a second number must stay within bounds, and the project may be a
shared repository where the loop should leave nothing of its own. These options cover that. All are
off by default.

```toml
[project]
name = "latency"
repo = "~/work/service"      # this config lives in ~/.rsi/latency/, not in the repository
base_branch = "main"

[setting]
benchmark = "latency"
model = "load-test"

[harness]
editable = ["src/**", "config/**"]

[tasks]
rollouts = 4
paired = true
confirm = true

[commands]
run_task = "python3 ~/tools/measure.py {task_file} {checkout} {out} {rollout}"

[budget]
rounds = 5
candidates = 3
parallel = 4
min_gain = 1.0

[[constraints]]
metric = "cost_usd"
max_increase_pct = 10

[[constraints]]
metric = "accuracy_pct"
max_decrease_pct = 0

[agents]
cli = "agy"
model = "gemini-4-neon"
effort = "high"

[promotion]
mode = "per_edit"
draft = true
branch_prefix = "rsi/"
```

**Metrics.** `result.json` may carry named numbers beside the score:
`{"score": 0.61, "metrics": {"p95_ms": 182, "cost_usd": 4.20}}`. Names are lower-case with digits and
underscores. A rollout need not report every metric. Their means are shown to the agents, in
`rsi status`, in the memory and in pull requests. The score is still the one thing the loop optimizes.

**Limits.** Each `[[constraints]]` entry names a metric and `max_increase_pct` or `max_decrease_pct`
(0 means not at all). A limit is relative to the version the run started from, and it adds up over the
rounds: an edit that would take the total past a limit cannot be accepted, whatever the selector says.
The same holds for a candidate that does not report a limited metric the current version reports. A
limit covers one run. To keep a total over several runs, lower it as earlier changes are merged.

**Paired measurement.** With `paired = true`, every round scores the current version again in the
same batch as the candidates, and compares them with that fresh result. A change in the machines
between rounds then is not mistaken for a gain or a loss. Held-out is scored once, at the end, for
the starting and the final version together. This costs one more set of task runs per round;
`rsi run --dry-run` shows the count. If a run of that fresh baseline fails, nothing is accepted that
round.

**Confirmation.** The candidate the selector picks is the best of several noisy measurements, so its
first numbers flatter it: with many candidates, one that changes nothing will now and then look like
a gain. With `confirm = true` (it needs `paired = true`) the chosen candidate and the current version
are measured once more, in a batch of their own, and only that second measurement decides. Its
numbers are the ones recorded. This costs two more sets of task runs per accepted round.

**The current version.** `run_task` sees `RSI_CURRENT=1` when it scores the version the loop holds
now (a version that passed before), and an empty `RSI_CURRENT` for a candidate. A scorer can treat a
failure of the current version as a failed measurement (exit non-zero) rather than a score of 0, so a
flaky failure of the baseline does not make every candidate look better.

**Minimum gain.** `min_gain` is the number of score points a candidate must gain to be accepted.

**Predictions.** In its manifest a proposer may give a `"title"` for the edit and, under
`"expected"`, the percent change it expects per metric: `{"title": "Cache the parsed config", "expected":
{"p95_ms": -20}}`. The memory keeps the expected and the measured change side by side.

**Files outside the repository.** `rsi init --external` writes the config to
`~/.rsi/<name>/config.toml` with `repo` set. The memory (`sim.json`) and the log (`log.md`) then live
beside it, the loop commits nothing of its own, and the run branch holds only the accepted edits. The
memory is one file for all runs of the project, whether or not their edits were merged.

**One pull request per edit.** With `mode = "per_edit"` each accepted edit gets its own branch,
`<branch_prefix><run>-r<round>`, and pull request. The edit is cherry-picked onto the commit the run
started from. When it does not apply there, it depends on earlier edits: its branch then contains
them, and its pull request says which to merge first. Each body gives the measurements before and
after, and what the edit was measured on top of. Pull requests are drafts when `draft = true`, when
the held-out score went down, or when the held-out total is past a limit. Promoting again opens only
what is missing.

**Agents.** `cli = "agy"` uses the `agy` CLI instead of `claude`. The two are kept in bounds
differently:

| | `claude` (default) | `agy` |
|---|---|---|
| Shell | removed by restricted mode | every command is denied |
| Files outside its folders | out of reach in restricted mode, and denied by rule | denied |
| Private directory | denied by rule | outside its folders, so denied |
| Feedback and history folders | read-only on disk, and edits denied by rule | read-only on disk |

A denied action ends an `agy` session at once. What it wrote before still counts; if that is not a
valid answer, the call is made once more. Any agent call that leaves no valid output is made once
more, except after a timeout.

**Notes for the agents.** `notes = "agent-notes.md"` under `[agents]` names a text file of standing
notes from you: what is off limits and why, what has been measured, where to look. A relative path is
relative to the config file's folder. The text is shown to the proposer and to the selector at both of
its steps. It is read once a round, before the proposers start, so an edit reaches the next round
without a restart. While agents work the file is read-only on disk, like the feedback and history
folders, so nothing an agent does can change what the selector is shown. A run does not start when the
file is missing, is not UTF-8 text, or is one the loop may edit; if it cannot be read later, the loop
warns once and goes on without it.

## Progress

While `rsi run` works it keeps one live line at the bottom of the terminal, rewritten every few
seconds, and prints a line that stays for every finished task run and agent call, so a failure shows
the moment it happens:

```
[rsi] 20261124-1818 round 2/5: measuring
[rsi] round 2 · measuring · s2 · task-7 run 3/4: score 0.53 in 13m11s
[rsi] round 2 · measuring · s2 · task-7 run 4/4 FAILED: exit code 3
[rsi] round 2 · proposer s1: done in 9m12s
round 2/5 · measuring · runs 9/16, 1 failed · agents - · score 61.20 · 38m in step, 4h12m total, ~7h05m left
```

What the live line says is a template in the config. This section is optional; these are the defaults:

```toml
[progress]
line = "round {round}/{rounds} · {step} · runs {done}/{total}, {failed} failed · agents {agents} · score {score} · {step_elapsed} in step, {elapsed} total, ~{eta} left"
events = "all"          # lines for finished task runs and agent calls: "all", "failures" or "none"
interval_s = 5          # how often the line is redrawn in a terminal
log_interval_s = 300    # how often it is printed as a plain line when the output is not a terminal
```

| Field | Meaning |
|---|---|
| `run`, `status` | Run id; `running`, or `error` after a step failed |
| `round`, `rounds` | Round being worked on, counted from 1 (0 before the first), and how many there are |
| `step` | What the loop is doing, in words: baseline, asking agents, screening, measuring, selecting, confirming, recording, held-out, opening pull requests |
| `done`, `total`, `failed` | Task runs of the current step: finished, planned, failed to run |
| `agents` | Agent calls of the current step, running/planned; `-` when none |
| `errors`, `accepted` | Failed task runs and agent calls so far; rounds whose change was kept |
| `score`, `start_score` | Current and starting score |
| `metric.<name>` | A metric of the current version, such as `{metric.p95_ms}` |
| `limit.<name>` | How much of a limit the kept changes have used, such as `{limit.cost_usd}` |
| `elapsed`, `step_elapsed`, `eta` | Active time of the run and of the step, and the time left |

A field without a value shows `-`; the time left shows `?` until a task run has finished. It counts
the most task runs the rest of the run can take, so it tends to be high. A field the template names
that does not exist, or a format spec that does not suit its field, is refused when the run starts. A
field takes a Python format spec: `{score:.1f}` and `{metric.p95_ms:.0f}` format the number, `{step:>12}`
pads the words. A project can use its own words, for example
`line = "build {done}/{total} · p95 {metric.p95_ms:.0f} ms · ~{eta} left"`.

When the output is not a terminal (a log file, `nohup`), the live line is printed as an ordinary line
every `log_interval_s` and at each step's start, so a log carries no control characters.

## Where things live

| Place | Contents |
|---|---|
| `.rsi/config.toml` in the project | The config |
| `.rsi/sim.json`, `.rsi/log.md` on the run branch | The memory and a per-round log of evolve scores |
| `~/.rsi/<name>/private/` | Tasks, the split, run state, held-out results, stored live outcomes. Denied to agents |
| `~/.rsi/<name>/runs/<run>/` | Evolve results, each round's candidates, agent prompts and logs |
| `~/.rsi/<name>/worktrees/<run>/` | Git worktrees. Your working copy is never touched |

Set `RSI_HOME` to use a directory other than `~/.rsi`.

## Guards

- While agents run, the feedback and history folders are read-only on disk, whatever agent is used.

- A candidate may only change `harness.editable` paths. `harness.frozen` and `.rsi/` are off limits.
  A `run_task`, `check` or `ingest` command may not name any file the loop can edit: freeze the file,
  or call it from a frozen script.
- The default agents run `claude` in restricted mode: file tools only, confined to their worktree and
  the run's feedback, history and output folders, with no shell, no MCP servers and none of your own
  settings. On top of that, every agent call denies the private directory and the task file, and makes
  the feedback and history folders read-only. A custom `[agents]` command gives up these guarantees;
  `rsi validate` warns about it. These are the CLI's permission rules, not an operating-system sandbox.
- After the selector runs, anything it changed in the run branch's worktree is discarded. Before a
  candidate is scored, its worktree is reset to exactly its commit, so a file git ignores cannot
  influence the score.
- `run_task` runs the candidate's code with your privileges, and the framework does not sandbox it. An
  edit could in principle read the task file it is handed or tamper with the scoring. Keep the scoring
  logic in frozen files, pass the harness only the task's input, and read the diff in the pull request.
- A run stops before optimizing when too many rollouts fail to run at all.
- Budgets cap rounds, candidates, parallel runs, timeouts and total active time.

## Tests

```bash
bin/test                          # everything, in under a minute
RSI_SMOKE=1 bin/test -k smoke     # with the real claude CLI (costs tokens): one round on the toy
                                  # project, and a check that the default agent is confined
```

`examples/toy/` is a tiny project used by the tests: a rules file as the harness, a deterministic
scorer, twelve tasks and a scripted agent.
