---
name: rsi
description: Put a project into a recursive self-improvement loop with the `rsi` CLI, or run and inspect a loop that is already set up. Use when the user wants a project "in the RSI loop", wants its benchmark score improved iteratively, asks about a run's status or memory, or types /rsi.
---

# RSI loops

`rsi` runs a loop on a project: score the current version on a task set, let agents propose edits,
keep at most one edit per round, and record in a memory file (`.rsi/sim.json`) why each kept edit was
made and what it measurably did. You do not run the loop yourself. You set a project up and call the
CLI. All loop logic is in the CLI.

Check the CLI is installed with `rsi --help`. If it is not, install it from the framework's repo:
`python3 -m pip install -e <path to sk_rsi>`.

## Setting up a project

Work in the project's repo. Ask the user these questions one at a time, and skip any the repo already
answers:

1. **What may the loop change?** These become `harness.editable` globs. Keep them narrow.
2. **What is one task?** One unit of work with a known right answer, for example one past pull
   request to review or one input with an expected output.
3. **How is a task scored from 0 to 1?** This becomes the `run_task` command.
4. **Which model executes the tasks?** This is a label, `setting.model`.
5. **Is there a fast check** (unit tests, lint) that a broken edit would fail? This becomes `check`.
6. **Are there live outcomes** worth feeding back (real results, complaints, incidents)? This becomes
   `ingest`. It is optional and can come later.

Then:

1. Run `rsi init`. It writes `.rsi/config.toml` with every option commented.
2. Edit the config with the answers.
3. Write the `run_task` command as a script in the repo. Its contract:
   - It receives `{task_file}` (one task as JSON), `{checkout}` (the version to score) and `{out}`.
   - It runs the project **from `{checkout}`**, not from the user's working copy.
   - It writes `{out}/result.json`: `{"score": <0..1>, "summary": "<what happened>"}`.
   - It must not change `{checkout}`, post anything, or touch production.
   - Whatever it writes to `{out}` is shown to the proposer for evolve tasks. Put in what a person
     would need to diagnose a failure. For a held-out task nothing is shown.
4. Add the scorer script and its tests to `harness.frozen`. `rsi validate` refuses a `run_task` that
   the loop could rewrite.
5. Help the user write the task file at the path `rsi init` printed. It stays **outside the repo**
   because it holds the expected answers. Each line is a JSON object with a unique `id`.
6. Commit `.rsi/config.toml` and the scorer on the base branch.
7. Run `rsi validate` and fix what it reports.
8. Run `rsi run --dry-run` and show the user the number of task runs. Say what one task run costs in
   time and money before they start. A real run can take hours.

Do not start `rsi run` without the user's go-ahead after they have seen the dry-run count.

## Running and inspecting

```bash
rsi run                # start a run, or resume the unfinished one
rsi run --no-promote   # as above, without pushing or opening a pull request
rsi status             # round, step, scores, pull request
rsi sim show           # the memory as readable text
rsi ingest             # pull live outcomes now
rsi continue           # fold queued live tasks into the benchmark, then run
rsi promote            # open the pull request for a finished run
```

A run works on its own branch, `rsi/<run-id>`, in worktrees outside the repo. It never touches the
user's working copy. When it ends it pushes that branch and opens a pull request, as a draft when the
held-out score went down.

When the user changes `setting.model` or the task set, the next run carries the memory over and
records which setting each earlier attempt was measured in. Nothing else is needed.

## Rules

- Never edit `.rsi/sim.json` or `.rsi/log.md` by hand. `rsi sim merge FROM INTO` is the only manual
  change to the memory.
- Never move the task file into the repo, and never paste held-out tasks or their results into the
  conversation with a proposer or into a file in the repo.
- Never merge a run's branch yourself. The pull request is the user's gate.
- If a run stopped on an error, read `rsi status`, fix the cause, and run `rsi run` again. It resumes.
  Use `rsi run --new` only when the user wants to abandon the stopped run.
