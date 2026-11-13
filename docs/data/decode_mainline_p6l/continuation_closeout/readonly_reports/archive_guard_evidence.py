"""Archive a finished cache-bit experiment, with raw audit and own-only commit."""
from pathlib import Path
import collections, csv, datetime, gzip, hashlib, io, json, math, os
import subprocess, sys, tarfile, time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/workspace/composable_kernel')
DEST = REPO / 'docs/data/decode_mainline_p6l/continuation_guard'
sys.path.insert(0, str(ROOT))
import runtime as r


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def main():
    session = read(ROOT / 'session.json')
    assert time.time() < session['deadline_epoch']
    assert read(ROOT / 'guard_controller_status.json')['state'] == 'complete'
    assert read(ROOT / 'guard_qualification_audit.json')['ok']
    assert read(ROOT / 'guard_build_isa_audit.json')['ok']
    assert read(ROOT / 'guard_source_commit.json')['all26_build_source_SHA_verified']
    selection = dict(kernel_adoption=False, accepted_stable50=0, accepted_attributable30=0, no_new_seed=True)
    rows = read(ROOT / 'guard_discovery_results.json')
    assert len(rows) == 10
    binary = read(ROOT / 'compile/build16/completed.json')['benchmark_sha256']
    epoch = read(ROOT / 'environment_epoch.json')['id']
    assert all(x['binary_sha256'] == binary and x['epoch'] == epoch for x in rows)
    jobs = [json.loads(x) for x in (ROOT / 'jobs.jsonl').read_text().splitlines()]
    jobs = [x for x in jobs if x['benchmark_sha256'] == binary]
    started = [json.loads(x)['record'] for x in (ROOT / 'started.jsonl').read_text().splitlines()]
    started = [x for x in started if x['benchmark_sha256'] == binary]
    assert len(jobs) == len(started) == 35 + 10 * 8
    keys = {x['key'] for x in jobs}
    assert len(keys) == len(jobs) and keys == {x['key'] for x in started}
    gates = {x['key']: x for x in map(json.loads, (ROOT / 'gates.jsonl').read_text().splitlines())}
    checks = {x['key']: x for x in map(json.loads, (ROOT / 'kernel_checks.jsonl').read_text().splitlines())}
    raw, versions, manifests, comparisons = {}, {}, {}, []
    for job in jobs:
        assert job['ok'] and job['rc'] == 0
        assert job['physical_gpu'] == 4 and job['epoch'] == epoch
        assert session['t0_epoch'] <= datetime.datetime.fromisoformat(job['at']).timestamp() < session['gpu_deadline_epoch']
        assert job['environment']['HIP_VISIBLE_DEVICES'] == job['environment']['CUDA_VISIBLE_DEVICES'] == '4'
        gate = gates[job['key']]
        assert gate['rc'] == 0 and gate['hcu'] == '0.0%' and 'MKFD' not in gate['output'].upper()
        assert gate['mem_available_kib'] >= 64 * 1024 * 1024
        assert all(checks[job['key'] + s]['rc'] == 0 and not checks[job['key'] + s]['events'] for s in ['_pre', '_post'])
        parsed = r.parse_output((Path(job['path']) / 'run.log').read_text(), job['kind'],
                                job.get('expected'), job.get('repeats', 1), job.get('regression_kind'))
        if job['kind'] == 'accuracy':
            assert [list(x) for x in parsed['comparisons']] == job['comparisons']
            assert all(int(x[1]) == math.prod(job['shape'][k] for k in ['M', 'N', 'G']) and int(x[2]) == 0
                       for x in parsed['comparisons'])
            comparisons += parsed['comparisons']
        if job['kind'] == 'perf':
            assert parsed['result'] == job['result']
        for path in Path(job['path']).iterdir():
            if path.is_file():
                assert path.suffix not in ['.perf', '.o', '.co', '.isa']
                raw['jobs/' + path.parent.name + '/' + path.name] = path.read_bytes()
        for name, digest in job['script_sha256'].items():
            versions[name, digest] = ROOT / 'source_versions' / (digest + '_' + name)
        manifests[job['plan_sha256']] = ROOT / 'manifests' / (job['plan_sha256'] + '.json')
    assert len(comparisons) == 159
    for (_, digest), path in versions.items():
        data = path.read_bytes()
        assert sha(data) == digest
        raw['source_versions/' + path.name] = data
    for digest, path in manifests.items():
        raw['manifests/' + path.name] = path.read_bytes()
        assert read(path)['benchmark_sha256'] == binary
        for name, expected in read(path)['runner_sha256'].items():
            if name.startswith('drafts/'):
                data = (ROOT / name).read_bytes()
                assert sha(data) == expected
                raw['auxiliary_sources/' + name] = data
    for name in ['gates.jsonl', 'started.jsonl', 'kernel_checks.jsonl']:
        lines = []
        for line in (ROOT / name).read_bytes().splitlines(keepends=True):
            item = json.loads(line)
            key = item['record']['key'] if name == 'started.jsonl' else item['key']
            if key in keys or (name == 'kernel_checks.jsonl' and key.rsplit('_', 1)[0] in keys):
                lines.append(line)
        raw['journals/' + name] = b''.join(lines)
    raw['journals/guard_controller_events.jsonl'] = (ROOT / 'guard_controller_events.jsonl').read_bytes()
    assert not DEST.exists()
    DEST.mkdir()
    names = ['session.json', 'authorization.json', 'initial_files_sha256.json', 'research_plan.json',
             'heldout_protocol.json', 'historical_exclusion_inventory.json', 'historical_final_attempts.json',
             'environment_epoch.json', 'recovery_total.json', 'compile/build16/source_sha256.json',
             'compile/build16/completed.json', 'readonly_reports/archive_guard_evidence.py',
             'readonly_reports/commit_guard_source.py', 'readonly_reports/policy_cpu_diagnosis/plan.json',
             'readonly_reports/policy_cpu_diagnosis/results.json', 'readonly_reports/policy_cpu_diagnosis/summary.json']
    names += [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob('guard_*')) if p.suffix in ['.json', '.py']]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'drafts/policy_fallback_guard').rglob('*')) if p.suffix in ['.json', '.py', '.patch'] and p.name != 'before_user.patch']
    for name in sorted(set(names)):
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (DEST / 'jobs.json.gz').write_bytes(gzip.compress(json.dumps(jobs).encode(), mtime=0))
    inventory = []
    with tarfile.open(DEST / 'raw_jobs.tar.gz', 'w:gz') as archive:
        for name, data in sorted(raw.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mtime = len(data), 0
            archive.addfile(member, io.BytesIO(data))
            inventory.append(dict(path=name, bytes=len(data), sha256=sha(data)))
    with tarfile.open(DEST / 'raw_jobs.tar.gz') as archive:
        assert len(archive.getmembers()) == len(inventory)
        for expected, member in zip(inventory, archive.getmembers()):
            assert member.name == expected['path'] and sha(archive.extractfile(member).read()) == expected['sha256']
    write(DEST / 'raw_manifest.json', inventory)
    pairs = {x['key']: x for x in read(ROOT / 'guard_discovery_plan.json')['discovery_pairs']}
    fields = ['pair', 'ID', 'M', 'N', 'K', 'G', 'variant', 'control', 'base_instance', 'candidate_instance',
              'CK_base_ms', 'CK_candidate_ms', 'CK_base_TFLOPS', 'CK_candidate_TFLOPS', 'frozen_BLAS_ms',
              'frozen_BLAS_TFLOPS', 'R_base_frozen', 'R_candidate_frozen', 'delta_pct', 'high_cv',
              'base_rounds', 'candidate_rounds', 'base_config', 'candidate_config', 'epoch', 'binary_sha256', 'keys']
    with (DEST / 'paired_results_with_tflops.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            value = {k: row[k] for k in fields if k in row}
            value.update(row['shape'])
            flops = 2 * math.prod(row['shape'][k] for k in ['M', 'N', 'K', 'G'])
            value.update(CK_base_TFLOPS=flops / row['CK_base_ms'] / 1e9,
                         CK_candidate_TFLOPS=flops / row['CK_candidate_ms'] / 1e9,
                         frozen_BLAS_TFLOPS=flops / row['frozen_BLAS_ms'] / 1e9,
                         R_base_frozen=row['frozen_BLAS_ms'] / row['CK_base_ms'],
                         R_candidate_frozen=row['frozen_BLAS_ms'] / row['CK_candidate_ms'],
                         base_config=pairs[row['pair']]['base'], candidate_config=pairs[row['pair']]['candidate'])
            for key in ['base_rounds', 'candidate_rounds', 'base_config', 'candidate_config', 'keys']:
                value[key] = json.dumps(value[key], ensure_ascii=False, separators=(',', ':'))
            writer.writerow(value)
    summary = dict(at=r.stamp(), ok=True, physical_gpu=4, epoch=epoch, binary_sha256=binary,
                   jobs=len(jobs), kinds=dict(collections.Counter(x['kind'] for x in jobs)),
                   qualification_jobs=35, three_round_pairs=10, full_comparisons=159,
                   full_elements=sum(int(x[1]) for x in comparisons), failed_elements=0,
                   new_SQTT_captures=0, all1766_shader_machinecodes_unchanged=True, selection=selection,
                   adopted=False, accepted_stable50=0, accepted_attributable30=0,
                   Goal_complete=False, GUI_verified=False)
    write(DEST / 'audit_summary.json', summary)
    (DEST / 'README.md').write_text('\n'.join([
        '# Decode 优先策略：减少无效回退检查', '',
        '该修复属于既有 host-policy 固定开销因素：当几何不满足任何请求的新 decode 契约时，跳过一次必然失败的优先 selector 调用，再执行原完整策略。没有新增流水、shape 特判、环境缓存、公开 ABI 或选核结果变化。', '',
        f'benchmark SHA256：`{binary}`。1766个GPU shader的完整机器码和资源记录与修复前相同。322845项CPU对照的status/instance/reason全部相同，包含全部64种public-disable组合。', '',
        '35项资格、10组三轮配对、159次全输出和真实guard全部通过。固定清单为7个原回退控制与620/621/805三条主线对照；全部配置、实际实例、轮次、TFLOPS、R_frozen与高CV保留。', '',
        '三条主线仍使用已被拒绝的6035候选，只用于检查host修复没有改变流水，不构成整域性能采用。没有抽取新的盲测seed，也没有把这10组或历史跨构建赢家拼入正式表。50/30计数仍为0/0。', '',
        'CPU抽取策略体的计时仅是辅助诊断，排除了架构检测、HIP启动和GPU；其绝对时间不能直接等同GPU事件耗时。旧/新GPU二进制的差值不冒充同二进制纯host修复收益。', '',
        '本阶段无需重复捕获机器码完全相同的SQTT；先前第15批仍为其自身二进制的诊断证据。正式benchmark包含默认selector/run适配器，include_args_build0没有移除host策略解析，短kernel时间需考虑这部分影响。', '',
        'P6k、冻结BLAS、1193shape和原Excel不变；只做本地实验提交，不push、不合入。编译输入26份保留现场，未将用户overlay源文件代为提交。', '',
    ]))
    write(DEST / 'manifest.json', [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size,
                                      sha256=sha(p.read_bytes())) for p in sorted(DEST.rglob('*')) if p.is_file()])
    before = git('diff')
    assert not git('diff', '--cached', '--name-only').strip()
    index = ROOT / 'guard_archive.index'
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
    subprocess.run(['git', 'commit', '-m', 'P6l host回退修复10对与CPU策略等价证据归档（实验用，不合入）'], cwd=REPO, env=env, check=True)
    for name, oid in owned.items():
        assert git('show', 'HEAD:' + name) == (REPO / name).read_bytes()
        subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + name], cwd=REPO, check=True)
    assert git('diff') == before and not git('diff', '--cached', '--name-only').strip()
    receipt = dict(at=r.stamp(), commit=git('rev-parse', 'HEAD').decode().strip(), files=len(owned),
                   raw_members=len(inventory), Git_blob_verified=True, user_worktree_preserved=True,
                   push=False, Goal_complete=False)
    write(ROOT / 'guard_archive_commit.json', receipt)
    print('NOSLC_RAW_AND_OWN_GIT_ARCHIVE_PASS', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
