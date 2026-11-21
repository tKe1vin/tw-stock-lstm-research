"""Real Git hooks must never publish bytes that were not validated."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import persist_daily_state as persist


@pytest.fixture
def checkpoint_repo(tmp_path, monkeypatch):
    root, remote = tmp_path / 'working', tmp_path / 'remote.git'
    root.mkdir()
    def git(*args, cwd=root):
        return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', '/dev/null')
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.delenv('GITHUB_REF_NAME', raising=False)
    git('init', '--bare', '--initial-branch=main', str(remote))
    git('init', '-b', 'main')
    git('config', 'user.name', 'State stress test')
    git('config', 'user.email', 'state-stress@example.invalid')
    (root / 'data').mkdir()
    (root / 'data/news_update_state.json').write_text('{"revision":"baseline"}\n')
    (root / 'source.py').write_text('original = True\n')
    git('add', '.')
    git('commit', '-m', 'baseline')
    git('remote', 'add', 'origin', str(remote))
    git('push', '-u', 'origin', 'main')
    monkeypatch.setattr(persist, 'ROOT', root)
    monkeypatch.setattr(persist, 'IMMUTABLE_SHA256', {})
    return root, remote, git


def hook(root, name, body):
    path = root / '.git/hooks' / name
    path.write_text('#!/bin/sh\n' + body + '\n')
    path.chmod(0o755)


@pytest.mark.parametrize('stage', ['pre-commit', 'post-commit'])
def test_hook_changed_selected_bytes_never_enter_remote(checkpoint_repo, stage):
    root, remote, git = checkpoint_repo
    state = root / 'data/news_update_state.json'
    state.write_text('{"revision":"validated"}\n')
    before = git('rev-parse', 'HEAD').stdout
    hook(root, stage, "printf '%s\\n' '{\"revision\":\"changed by hook\"}' > data/news_update_state.json\ngit add -- data/news_update_state.json")
    with pytest.raises(ValueError, match='bytes changed'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'HEAD').stdout == before
    assert git('rev-parse', 'main', cwd=remote).stdout == before
    assert not git('diff', '--cached', '--name-only').stdout
    assert 'changed by hook' in state.read_text()


def test_successful_hook_unrelated_staging_is_preserved_for_review(checkpoint_repo):
    root, remote, git = checkpoint_repo
    (root / 'data/news_update_state.json').write_text('{"revision":"validated"}\n')
    before = git('rev-parse', 'HEAD').stdout
    hook(root, 'pre-commit', "printf '%s\\n' 'hook code' > source.py\ngit add -- source.py")
    with pytest.raises(ValueError, match='unrelated or unselected'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'HEAD').stdout == before
    assert git('rev-parse', 'main', cwd=remote).stdout == before
    assert git('diff', '--cached', '--name-only').stdout == b'source.py\n'
    assert (root / 'source.py').read_text() == 'hook code\n'


def test_pending_checkpoint_must_preserve_remote_attempt_baseline(checkpoint_repo):
    root, remote, git = checkpoint_repo
    name = 'reports/attempts/2026-11-21/20261121T170000000000+0800_0123456789abcdef.json'
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"published"}\n')
    git('add', name)
    git('commit', '-m', 'publish attempt')
    git('push', 'origin', 'main')
    remote_head = git('rev-parse', 'HEAD').stdout
    path.write_text('{"status":"rewritten"}\n')
    git('commit', '-am', 'research: daily state 2026-11-22 [skip ci]')
    unpublished_head = git('rev-parse', 'HEAD').stdout
    with pytest.raises(ValueError, match='Published daily attempt changed'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'HEAD').stdout == unpublished_head
    assert git('rev-parse', 'main', cwd=remote).stdout == remote_head


def test_unknown_tracking_ref_rejected_before_new_checkpoint(checkpoint_repo):
    root, remote, git = checkpoint_repo
    before = git('rev-parse', 'HEAD').stdout
    (root / 'data/news_update_state.json').write_text('{"revision":"new"}\n')
    git('update-ref', '-d', 'refs/remotes/origin/main')
    with pytest.raises(RuntimeError, match='known remote-tracking'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'HEAD').stdout == before
    assert git('rev-parse', 'main', cwd=remote).stdout == before
    assert not git('diff', '--cached', '--name-only').stdout


def test_hook_additional_commit_is_preserved_but_not_published(checkpoint_repo):
    root, remote, git = checkpoint_repo
    before = git('rev-parse', 'HEAD').stdout
    (root / 'data/news_update_state.json').write_text('{"revision":"validated"}\n')
    hook(root, 'post-commit', 'mv .git/hooks/post-commit .git/hooks/post-commit.done\n'
         "printf '%s\\n' 'hook-owned work' > source.py\ngit add -- source.py\ngit commit -m hook-owned-work")
    with pytest.raises(ValueError, match='HEAD changed beyond'):
        persist.main(['--commit', '--push'])
    assert git('show', '-s', '--format=%s').stdout.strip() == b'hook-owned-work'
    assert git('rev-parse', 'main', cwd=remote).stdout == before
    assert (root / 'source.py').read_text() == 'hook-owned work\n'


def advance_remote(checkpoint_repo):
    root, remote, git = checkpoint_repo
    peer = root.parent / 'peer'
    git('clone', str(remote), str(peer))
    git('config', 'user.name', 'Remote progress', cwd=peer)
    git('config', 'user.email', 'peer@example.invalid', cwd=peer)
    (peer / 'README.md').write_text('Concurrent legitimate remote progress.\n')
    git('add', 'README.md', cwd=peer)
    git('commit', '-m', 'remote progress', cwd=peer)
    git('push', 'origin', 'main', cwd=peer)
    return git('rev-parse', 'main', cwd=remote).stdout


def test_post_rewrite_hook_commit_cannot_enter_remote_after_a_successful_rebase(checkpoint_repo):
    root, remote, git = checkpoint_repo
    remote_head = advance_remote(checkpoint_repo)
    (root / 'data/news_update_state.json').write_text('{"revision":"validated"}\n')
    hook(root, 'post-rewrite', 'mv .git/hooks/post-rewrite .git/hooks/post-rewrite.done\n'
         "printf '%s\\n' 'hook_owned = True' > source.py\ngit add -- source.py\ngit commit -m hook-owned-after-rebase")
    with pytest.raises(RuntimeError, match='unrelated|Unpublished'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'main', cwd=remote).stdout == remote_head
    assert git('show', '-s', '--format=%s').stdout.strip() == b'hook-owned-after-rebase'
    assert (root / 'source.py').read_text() == 'hook_owned = True\n'
    assert (root / 'data/news_update_state.json').read_text() == '{"revision":"validated"}\n'
    assert not (root / '.git/rebase-merge').exists()


def test_post_rewrite_hook_staging_is_preserved_but_blocks_the_final_push(checkpoint_repo):
    root, remote, git = checkpoint_repo
    remote_head = advance_remote(checkpoint_repo)
    (root / 'data/news_update_state.json').write_text('{"revision":"validated"}\n')
    hook(root, 'post-rewrite', 'mv .git/hooks/post-rewrite .git/hooks/post-rewrite.done\n'
         "printf '%s\\n' 'hook_staged = True' > source.py\ngit add -- source.py")
    with pytest.raises((RuntimeError, ValueError), match='staged|dirty|changed|rebase'):
        persist.main(['--commit', '--push'])
    assert git('rev-parse', 'main', cwd=remote).stdout == remote_head
    assert git('diff', '--cached', '--name-only').stdout == b'source.py\n'
    assert (root / 'source.py').read_text() == 'hook_staged = True\n'
    assert (root / 'data/news_update_state.json').read_text() == '{"revision":"validated"}\n'
