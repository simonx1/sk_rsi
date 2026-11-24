from rsi import project
from rsi.config import ConfigError
from rsi.workspace import GitError
from tests.toy import ToyCase


class FindTest(ToyCase):
    def test_in_repo_config(self):
        repo = self.make_toy()
        found = project.find(repo)
        self.assertEqual((found.repo, found.external, found.cfg.name), (repo, False, "toy"))
        self.assertEqual(found.config_path, repo / ".rsi" / "config.toml")
        self.assertEqual(found.sim_path, repo / ".rsi" / "sim.json")
        self.assertEqual(found.layout.root, self.home / "toy")

    def test_external_config(self):
        repo = self.make_toy(external=True)
        self.assertFalse((repo / ".rsi").exists())
        found = project.find(repo)
        self.assertEqual((found.repo, found.external, found.cfg.name), (repo, True, "toy"))
        self.assertEqual(found.config_path, self.home / "toy" / "config.toml")
        self.assertEqual(found.sim_path, self.home / "toy" / "sim.json")

    def test_found_from_a_subdirectory(self):
        repo = self.make_toy(external=True)
        (repo / "sub").mkdir()
        self.assertEqual(project.find(repo / "sub").repo, repo)

    def assert_error(self, repo, *fragments):
        with self.assertRaises(ConfigError) as caught:
            project.find(repo)
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_repo_key_inside_the_repository_is_an_error(self):
        repo = self.make_toy(project_extra='repo = "~/somewhere"')
        self.assert_error(repo, "project.repo is only for a config kept outside the repository")

    def test_two_external_configs_for_one_repository(self):
        repo = self.make_toy(external=True)
        other = self.home / "other" / "config.toml"
        other.parent.mkdir()
        other.write_text((self.home / "toy" / "config.toml").read_text().replace('name = "toy"', 'name = "other"'))
        self.assert_error(repo, str(other), str(self.home / "toy" / "config.toml"))

    def test_folder_name_must_match_project_name(self):
        repo = self.make_toy(external=True)
        (self.home / "toy").rename(self.home / "renamed")
        self.assert_error(repo, "project.name", "renamed")

    def test_no_config(self):
        repo = self.make_toy(external=True)
        (self.home / "toy" / "config.toml").unlink()
        self.assert_error(repo, ".rsi/config.toml", str(self.home))

    def test_an_unreadable_config_of_another_project_is_ignored(self):
        repo = self.make_toy(external=True)
        broken = self.home / "other" / "config.toml"
        broken.parent.mkdir()
        broken.write_text("[project\n")
        self.assertEqual(project.find(repo).cfg.name, "toy")

    def test_a_broken_config_in_the_repository_is_an_error(self):
        repo = self.make_toy()
        (repo / ".rsi" / "config.toml").write_text("[project\n")
        self.assert_error(repo, "not valid TOML")

    def test_outside_a_git_repository(self):
        with self.assertRaises(GitError):
            project.find(self.tmp)
