"""Commit the owned adapter insertion, preserving every user edit."""
from pathlib import Path
import difflib, hashlib, json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plan, runtime as r
REPO = plan.REPO
D = ROOT / 'drafts/policy_fallback_guard'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def edits(a, b):
    aa, bb = a.splitlines(keepends=True), b.splitlines(keepends=True)
    return [(aa[i:j], bb[k:l]) for op, i, j, k, l in difflib.SequenceMatcher(None, aa, bb, autojunk=False).get_opcodes() if op != 'equal']


def main():
    assert r.read(ROOT / 'guard_controller_status.json')['state'] == 'complete'
    assert r.read(ROOT / 'guard_discovery_analysis.json')['complete_phase']
    assert r.read(ROOT / 'guard_qualification_audit.json')['ok']
    pins = r.read(ROOT / 'compile/build16/source_sha256.json')
    assert all(plan.sha(REPO / name) == digest for name, digest in pins.items())
    decision = r.read(D / 'plan.json')
    name = decision['source']
    before_head = (D / 'before_HEAD.cpp').read_text()
    before_work = (D / 'before_worktree.cpp').read_text()
    after_head = (D / 'after_HEAD.cpp').read_text()
    after_work = (REPO / name).read_text()
    assert git('show', 'HEAD:' + name).decode() == before_head
    assert plan.sha(REPO / name) == decision['after_worktree_sha256']
    assert hashlib.sha256(after_head.encode()).hexdigest() == decision['after_HEAD_sha256']
    assert edits(before_head, before_work) == edits(after_head, after_work), 'Uncommitted user change chunks must remain exact'
    initial = r.read(ROOT / 'initial_files_sha256.json')
    user_names = [x[3:] for x in (ROOT / 'initial_status.txt').read_text().splitlines() if not x.startswith('?? ') and x[3:] != name]
    old_diff = git('diff', '--', *user_names)
    assert not git('diff', '--cached', '--name-only').strip()
    index = ROOT / 'guard_source.index'
    assert not index.exists()
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(index)
    subprocess.run(['git', 'read-tree', 'HEAD'], cwd=REPO, env=env, check=True)
    blob = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], cwd=REPO, input=after_head.encode()).decode().strip()
    subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name], cwd=REPO, env=env, check=True)
    assert git('diff', '--cached', '--name-only', env=env).decode().splitlines() == [name]
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPO, env=env, check=True)
    subprocess.run(['git', 'commit', '-m', 'P6l 避免不适用decode优先选核的额外回退开销（实验用，不合入）'], cwd=REPO, env=env, check=True)
    assert git('show', 'HEAD:' + name).decode() == after_head
    subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name], cwd=REPO, check=True)
    assert not git('diff', '--cached', '--name-only').strip()
    assert (REPO / name).read_text() == after_work and git('diff', '--', *user_names) == old_diff
    assert edits(git('show', 'HEAD:' + name).decode(), (REPO / name).read_text()) == edits(before_head, before_work)
    untracked = [x[3:] for x in (ROOT / 'initial_status.txt').read_text().splitlines() if x.startswith('?? ')]
    assert len(untracked) == 10 and all(plan.sha(REPO / n) == initial[n] for n in untracked)
    assert hashlib.md5(git('diff', '--', 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp')).hexdigest() == '4c7f49a5c7592767d53195a713db6b8a'
    receipt = dict(at=r.stamp(), commit=git('rev-parse', 'HEAD').decode().strip(), files=[name],
                   user_change_chunks_exact=True, other_user_tracked_diff_bytes_exact=True,
                   user_untracked10_SHA_verified=True, readfirstlane_preserved=True,
                   all26_build_source_SHA_verified=True, clean_owned_HEAD_blob_verified=True,
                   default_off=True, GPU_shaders_unchanged=1766, kernel_adoption=False,
                   Goal_complete=False, push=False)
    r.write(ROOT / 'guard_source_commit.json', receipt)
    print('GUARD_OWN_SOURCE_COMMIT_USER_CHUNKS_PRESERVED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
