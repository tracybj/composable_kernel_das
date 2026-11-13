"""Verify derived tables and commit only this new local closeout folder."""
from pathlib import Path
import csv, datetime, hashlib, json, math, os, subprocess
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/workspace/composable_kernel')
DEST = REPO / 'docs/data/decode_mainline_p6l/continuation_closeout'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def main():
    manifest = json.loads((DEST / 'manifest.json').read_text())
    for item in manifest:
        path = DEST / item['path']
        assert path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    with (DEST / 'projection_P6k_UNCHANGED_NOT_FRESH1193.csv').open(encoding='utf-8-sig') as stream:
        projection = list(csv.DictReader(stream))
    assert len(projection) == 1193 and {int(x['ID']) for x in projection} == set(range(1, 1194))
    for row in projection:
        flops = 2 * math.prod(int(row[k]) for k in ['M', 'N', 'K', 'G'])
        assert math.isclose(float(row['R_frozen']), float(row['frozen_BLAS_ms']) / float(row['CK_ms']), rel_tol=1e-12)
        assert math.isclose(float(row['CK_TFLOPS']), flops / float(row['CK_ms']) / 1e9, rel_tol=1e-12)
        assert math.isclose(float(row['frozen_BLAS_TFLOPS']), flops / float(row['frozen_BLAS_ms']) / 1e9, rel_tol=1e-12)
        assert row['adopted_replacement'] == row['fresh_1193_retest'] == 'False'
    with (DEST / 'fresh_buckets_by_stage_variant_NOT_ADOPTED.csv').open(encoding='utf-8-sig') as stream:
        buckets = list(csv.DictReader(stream))
    assert len(buckets) == 28
    for row in buckets:
        assert sum(int(value) for name, value in row.items() if name.startswith('R_')) == int(row['cases'])
        assert row['adopted'] == 'False'
    assert json.loads((ROOT / 'research_closed.json').read_text())['pending_GPU_jobs'] == 0
    assert not git('diff', '--cached', '--name-only').strip()
    before = git('diff')
    index = ROOT / 'continuation_closeout.index'
    assert not index.exists()
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(index)
    subprocess.run(['git', 'read-tree', 'HEAD'], cwd=REPO, env=env, check=True)
    owned = {}
    for path in sorted(DEST.rglob('*')):
        if path.is_file():
            name = str(path.relative_to(REPO))
            oid = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], cwd=REPO, input=path.read_bytes()).decode().strip()
            owned[name] = oid
            subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + name], cwd=REPO, env=env, check=True)
    assert set(git('diff', '--cached', '--name-only', env=env).decode().splitlines()) == set(owned)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPO, env=env, check=True)
    subprocess.run(['git', 'commit', '-m', 'P6l 新增窗口348对完整审计收尾，性能目标未达成（实验用，不合入）'], cwd=REPO, env=env, check=True)
    for name, oid in owned.items():
        assert git('show', 'HEAD:' + name) == (REPO / name).read_bytes()
        subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + name], cwd=REPO, check=True)
    assert git('diff') == before and not git('diff', '--cached', '--name-only').strip()
    initial_status = (ROOT / 'initial_status.txt').read_bytes()
    assert git('status', '--short', '--untracked-files=all') == initial_status
    receipt = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), commit=git('rev-parse', 'HEAD').decode().strip(),
                   files=len(owned), manifest_members_SHA_and_Git_blobs_verified=True,
                   all1193_projection_rows_numeric_and_source_flags_verified=True,
                   all28_fresh_bucket_rows_sum_verified=True, initial19_status_entries_unchanged=True,
                   user_worktree_diff_bytes_unchanged=True, no_pending_GPU_jobs=True,
                   performance_Goal_complete=False, push=False)
    (ROOT / 'continuation_closeout_commit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('CONTINUATION_OWN_CLOSEOUT_COMMIT_PASS', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
