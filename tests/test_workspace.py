import tempfile
import unittest
from pathlib import Path

from rsi import workspace
from tests.helpers import git, make_repo


class GlobTest(unittest.TestCase):
    def test_patterns(self):
        cases = [
            ("skill/**", "skill/SKILL.md", True),
            ("skill/**", "skill/scripts/a/b.py", True),
            ("skill/**", "skillful/x.md", False),
            ("skill/scripts/test_*.py", "skill/scripts/test_x.py", True),
            ("skill/scripts/test_*.py", "skill/scripts/sub/test_x.py", False),
            ("**/*.md", "README.md", True),
            ("**/*.md", "a/b/c.md", True),
            ("*.md", "a/b.md", False),
            ("rules.txt", "rules.txt", True),
            ("rules.txt", "rules_txt", False),
            ("a?.py", "ab.py", True),
            ("a?.py", "a/.py", False),
        ]
        for pattern, path, expected in cases:
            self.assertEqual(workspace.matches_any(path, [pattern]), expected, (pattern, path))

    def test_path_violations(self):
        found = workspace.path_violations(
            ["skill/SKILL.md", "skill/scripts/test_a.py", "README.md", ".rsi/sim.json", ".rsi"],
            editable=["skill/**"],
            frozen=["skill/scripts/test_*.py"],
        )
        self.assertEqual(found, [
            "skill/scripts/test_a.py: frozen",
            "README.md: not editable",
            ".rsi/sim.json: only the framework writes .rsi/",
            ".rsi: only the framework writes .rsi/",
        ])


class RepoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.repo = make_repo(self.tmp / "repo", {
            "rules.txt": "a => b\n",
            "frozen.py": "x = 1\n",
            ".gitignore": "*.log\n",
            "bin/score": "#!/bin/sh\n",
            ".rsi/config.toml": "[project]\n",
        })
        self.wt = self.tmp / "wt"

    def test_repo_root_and_rev_parse(self):
        self.assertEqual(workspace.repo_root(self.repo / "bin"), self.repo)
        self.assertEqual(len(workspace.rev_parse(self.repo)), 40)
        with self.assertRaises(workspace.GitError):
            workspace.repo_root(self.tmp)

    def test_show_reads_a_file_at_a_ref(self):
        (self.repo / "rules.txt").write_text("uncommitted\n")
        self.assertEqual(workspace.show(self.repo, "main", "rules.txt"), "a => b")
        with self.assertRaises(workspace.GitError):
            workspace.show(self.repo, "main", "missing.txt")

    def test_worktree_on_a_new_branch(self):
        workspace.add_worktree(self.repo, self.wt, "rsi/run1", "main")
        self.assertTrue((self.wt / "rules.txt").is_file())
        self.assertTrue(workspace.branch_exists(self.repo, "rsi/run1"))
        self.assertEqual(workspace.rev_parse(self.wt), workspace.rev_parse(self.repo, "main"))

    def test_worktree_can_be_re_added_for_an_existing_branch(self):
        workspace.add_worktree(self.repo, self.wt, "rsi/run1", "main")
        (self.wt / "rules.txt").write_text("c => d\n")
        sha = workspace.commit_all(self.wt, "edit")
        workspace.remove_worktree(self.repo, self.wt)
        self.assertFalse(self.wt.exists())
        workspace.add_worktree(self.repo, self.wt, "rsi/run1", "main")
        self.assertEqual(workspace.rev_parse(self.wt), sha)

    def test_reset_worktree_discards_everything(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        (self.wt / "rules.txt").write_text("c => d\n")
        workspace.commit_all(self.wt, "edit")
        (self.wt / "scratch").write_text("junk")
        workspace.reset_worktree(self.repo, self.wt, "cand/x", "main")
        self.assertEqual((self.wt / "rules.txt").read_text(), "a => b\n")
        self.assertFalse((self.wt / "scratch").exists())
        self.assertEqual(workspace.rev_parse(self.wt), workspace.rev_parse(self.repo, "main"))

    def test_remove_and_delete_are_safe_to_repeat(self):
        workspace.remove_worktree(self.repo, self.wt)
        workspace.delete_branch(self.repo, "never-existed")
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        workspace.remove_worktree(self.repo, self.wt)
        workspace.remove_worktree(self.repo, self.wt)
        workspace.delete_branch(self.repo, "cand/x")
        self.assertFalse(workspace.branch_exists(self.repo, "cand/x"))

    def test_commit_all_returns_none_when_nothing_changed(self):
        workspace.add_worktree(self.repo, self.wt, "rsi/run1", "main")
        self.assertIsNone(workspace.commit_all(self.wt, "nothing"))

    def test_commit_all_can_be_limited_to_paths(self):
        workspace.add_worktree(self.repo, self.wt, "rsi/run1", "main")
        (self.wt / ".rsi" / "sim.json").write_text("{}\n")
        (self.wt / "stray.pyc").write_text("left by a command")
        sha = workspace.commit_all(self.wt, "record", paths=[".rsi"])
        self.assertEqual(workspace.changed_files(self.repo, "main", sha), [".rsi/sim.json"])

    def test_changed_files_and_diff(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        (self.wt / "rules.txt").write_text("a => b\nc => d\n")
        (self.wt / "new.txt").write_text("n\n")
        sha = workspace.commit_all(self.wt, "edit")
        self.assertEqual(workspace.changed_files(self.repo, "main", sha), ["new.txt", "rules.txt"])
        self.assertIn("+c => d", workspace.diff_text(self.repo, "main", sha))

    def test_rename_reports_both_paths(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        git(self.wt, "mv", "frozen.py", "rules2.txt")
        sha = workspace.commit_all(self.wt, "rename a frozen file")
        files = workspace.changed_files(self.repo, "main", sha)
        self.assertEqual(files, ["frozen.py", "rules2.txt"])
        self.assertEqual(
            workspace.path_violations(files, editable=["rules*.txt"], frozen=["frozen.py"]),
            ["frozen.py: frozen"],
        )

    def test_non_ascii_paths_are_reported_literally(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        (self.wt / "frozen\u00e9.py").write_text("x = 2\n")
        sha = workspace.commit_all(self.wt, "add a file with an accent in its name")
        files = workspace.changed_files(self.repo, "main", sha)
        self.assertEqual(files, ["frozen\u00e9.py"])
        self.assertEqual(
            workspace.path_violations(files, editable=["**"], frozen=["frozen\u00e9.py"]),
            ["frozen\u00e9.py: frozen"],
        )

    def test_clean_discards_edits_untracked_and_ignored_files(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        self.assertEqual(workspace.clean(self.wt), [])
        (self.wt / "rules.txt").write_text("tampered\n")
        (self.wt / "stray.txt").write_text("x")
        (self.wt / "override.log").write_text("ignored by git, read by a scorer")
        dirty = workspace.clean(self.wt)
        self.assertEqual(len(dirty), 3)
        self.assertEqual((self.wt / "rules.txt").read_text(), "a => b\n")
        self.assertFalse((self.wt / "stray.txt").exists())
        self.assertFalse((self.wt / "override.log").exists())
        self.assertEqual(workspace.clean(self.wt), [])

    def test_cherry_pick_clean_and_conflicting(self):
        workspace.add_worktree(self.repo, self.wt, "edits", "main")
        (self.wt / "rules.txt").write_text("A => b\n")
        a = workspace.commit_all(self.wt, "A: change rules.txt")
        (self.wt / "frozen.py").write_text("x = 2\n")
        b = workspace.commit_all(self.wt, "B: change frozen.py")
        (self.wt / "rules.txt").write_text("AA => b\n")
        c = workspace.commit_all(self.wt, "C: change the line A changed")
        scratch = self.tmp / "scratch"
        workspace.add_detached_worktree(self.repo, scratch, "main")
        self.assertTrue(workspace.cherry_pick(scratch, b))
        self.assertEqual((scratch / "frozen.py").read_text(), "x = 2\n")
        self.assertEqual((scratch / "rules.txt").read_text(), "a => b\n")
        head = workspace.rev_parse(scratch)
        self.assertFalse(workspace.cherry_pick(scratch, c))
        self.assertEqual(git(scratch, "status", "--porcelain"), "")
        self.assertEqual(workspace.rev_parse(scratch), head)
        workspace.set_branch(self.repo, "picked", head)
        workspace.set_branch(self.repo, "picked", a)
        self.assertEqual(workspace.rev_parse(self.repo, "picked"), a)

    def test_a_worktree_whose_git_file_was_replaced_is_refused(self):
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        self.assertTrue(workspace.read_gitfile(self.wt).startswith("gitdir: "))
        (self.wt / ".git").unlink()
        (self.wt / "fake").mkdir()
        (self.wt / ".git").write_text("gitdir: fake\n")
        for call in (lambda: workspace.clean(self.wt), lambda: workspace.commit_all(self.wt, "x"),
                     lambda: workspace.fast_forward(self.wt, "HEAD"), lambda: workspace.cherry_pick(self.wt, "HEAD")):
            with self.assertRaises(workspace.GitError) as caught:
                call()
            self.assertIn(".git", str(caught.exception))
        (self.wt / ".git").unlink()
        (self.wt / ".git").symlink_to(self.repo / ".git")
        with self.assertRaises(workspace.GitError):
            workspace.clean(self.wt)

    def test_fast_forward_and_is_merged(self):
        main_wt = self.tmp / "main-wt"
        workspace.add_worktree(self.repo, main_wt, "rsi/run1", "main")
        workspace.add_worktree(self.repo, self.wt, "cand/x", "main")
        (self.wt / "rules.txt").write_text("c => d\n")
        sha = workspace.commit_all(self.wt, "edit")
        self.assertFalse(workspace.is_merged(self.repo, "cand/x", "rsi/run1"))
        workspace.fast_forward(main_wt, sha)
        workspace.fast_forward(main_wt, sha)
        self.assertEqual(workspace.rev_parse(main_wt), sha)
        self.assertTrue(workspace.is_merged(self.repo, "cand/x", "rsi/run1"))

    def test_command_in_editable(self):
        check = lambda command, editable, frozen=(): workspace.command_in_editable(
            command, self.repo, editable, frozen)
        self.assertTrue(check("bin/score {task_file} {out}", ["bin/**"]))
        self.assertFalse(check("bin/score {task_file} {out}", ["bin/**"], ["bin/score"]))
        self.assertFalse(check("bin/score {task_file} {out}", ["rules.txt"]))
        self.assertFalse(check("python3 score.py", ["**"]))
        self.assertTrue(check(f"{self.repo}/bin/score x", ["bin/**"]))
        # Fail closed: an editable file named anywhere in the command counts, whatever runs it.
        for command in (
            "python3 bin/score x",
            "env MODE=1 python3 -u bin/score x",
            "timeout 60 bin/score x",
            "true && bin/score x",
            "true; bin/score x",
            "bin/../bin/score x",
            "bash -c 'bin/score x'",
            "SCORER=bin/score run-it",
            "grep -q a bin/score",
        ):
            self.assertTrue(check(command, ["bin/**"]), command)
            self.assertFalse(check(command, ["bin/**"], ["bin/score"]), command)
        self.assertFalse(check("ls bin", ["bin/**"]))  # a directory is not a file the loop edits
        self.assertFalse(check("", ["**"]))
        self.assertFalse(check("echo 'unbalanced", ["**"]))

    def test_remotes(self):
        self.assertEqual(workspace.remotes(self.repo), [])
        git(self.repo, "remote", "add", "origin", "https://example.test/x.git")
        self.assertEqual(workspace.remotes(self.repo), ["origin"])
