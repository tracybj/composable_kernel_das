"""Reconstruct initial dirty files in a private index; compare user edit chunks."""
from pathlib import Path
import datetime, difflib, hashlib, json, os, subprocess, re
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/workspace/composable_kernel')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def edits(a, b):
    aa, bb = a.splitlines(keepends=True), b.splitlines(keepends=True)
    return [(aa[i:j], bb[k:l]) for op, i, j, k, l in difflib.SequenceMatcher(None, aa, bb, autojunk=False).get_opcodes() if op != 'equal']


def main():
    session = json.loads((ROOT / 'session.json').read_text())
    initial = json.loads((ROOT / 'initial_files_sha256.json').read_text())
    lines = (ROOT / 'initial_status.txt').read_text().splitlines()
    tracked = [s[3:] for s in lines if not s.startswith('?? ')]
    untracked = [s[3:] for s in lines if s.startswith('?? ')]
    assert len(tracked) == 9 and len(untracked) == 10
    index = ROOT / 'readonly_reports/initial_user_reconstruction.index'
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(index)
    # Reuse only if it still exactly reconstructs the same frozen initial tree.
    if not index.exists():
        subprocess.run(['git', 'read-tree', session['head']], cwd=REPO, env=env, check=True)
        subprocess.run(['git', 'apply', '--cached', str(ROOT / 'initial_tracked.patch')], cwd=REPO, env=env, check=True)
    rows = []; counter_adjustments = []
    for name in tracked:
        base = git('show', session['head'] + ':' + name)
        dirty = git('show', ':' + name, env=env)
        assert sha(dirty) == initial[name]
        current_head = git('show', 'HEAD:' + name)
        current_work = (REPO / name).read_bytes()
        before, after = edits(base, dirty), edits(current_head, current_work)
        if before != after:
            assert name in ['example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_query_test.cpp', 'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp']
            assert len(before) == len(after)
            mismatches = [(x, y) for x, y in zip(before, after) if x != y]
            assert len(mismatches) == 1
            old_pair, new_pair = mismatches[0]
            assert all(len(x) == 1 for x in [*old_pair, *new_pair])
            texts = [x[0] for x in [*old_pair, *new_pair]]
            assert all(b'std::array<' in x for x in texts)
            numbers = [int(re.search(rb', (\d+)>', x)[1]) for x in texts]
            assert numbers == [61, 63, 69, 71]
            normalized = [re.sub(rb', \d+>', b', COUNT>', x) for x in texts]
            assert len(set(normalized)) == 1
            assert numbers[1] - numbers[0] == numbers[3] - numbers[2] == 2
            assert numbers[2] - numbers[0] == numbers[3] - numbers[1] == 8
            counter_adjustments.append(dict(path=name, initial_base=61, initial_with_user=63, current_base=69, current_with_user=71, user_delta_preserved=2, owned_new_entries=8, all_other_changed_chunks_byte_exact=True))
        else:
            assert before == after
        fingerprint = sha(repr(before).encode())
        rows.append(dict(path=name, user_edit_chunks=len(before), user_edit_chunks_sha256=fingerprint,
                         user_candidate_entries_and_other_changed_lines_preserved=True))
    assert all(sha((REPO / name).read_bytes()) == initial[name] for name in untracked)
    rf = git('diff', '--', 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp')
    assert hashlib.md5(rf).hexdigest() == session['protected_diff_md5']
    assert not git('diff', '--cached', '--name-only').strip()
    report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), ok=True, CPU_only=True, GPU=False,
                  tracked9_user_changes_preserved=True, exact_registry_counter_adjustments=counter_adjustments, untracked10_exact_SHA=True, readfirstlane_diff_MD5=session['protected_diff_md5'],
                  initial_tree=session['head'], initial_patch_sha256=sha((ROOT / 'initial_tracked.patch').read_bytes()),
                  current_HEAD=git('rev-parse', 'HEAD').decode().strip(), private_index_only=True,
                  main_index_unstaged=True, rows=rows)
    (ROOT / 'readonly_reports/user_preservation_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('INITIAL_USER9_CHANGED_CHUNKS_AND10_UNTRACKED_SHA_PRESERVED', flush=True)


if __name__ == '__main__':
    main()
