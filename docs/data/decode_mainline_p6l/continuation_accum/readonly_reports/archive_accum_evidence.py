"""Archive a finished cache-bit experiment, with raw audit and own-only commit."""
from pathlib import Path
import collections, csv, datetime, gzip, hashlib, io, json, math, os
import subprocess, sys, tarfile, time

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/workspace/composable_kernel')
DEST = REPO / 'docs/data/decode_mainline_p6l/continuation_accum'
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
    assert read(ROOT / 'accum_controller_status.json')['state'] == 'complete'
    assert read(ROOT / 'accum_qualification_audit.json')['ok']
    assert read(ROOT / 'accum_build_isa_audit.json')['ok']
    assert read(ROOT / 'accum_source_commit.json')['source26_SHA_verified']
    selection = read(ROOT / 'accum_candidate_selection.json')
    # This archive never bypasses or substitutes for independent confirmation.
    assert not (ROOT / 'final_active.json').exists(), 'Do not archive unreviewed final results as discovery-only'
    rows = read(ROOT / 'accum_discovery_results.json')
    assert len(rows) == 23
    binary = read(ROOT / 'compile/build17/completed.json')['benchmark_sha256']
    epoch = read(ROOT / 'environment_epoch.json')['id']
    assert all(x['binary_sha256'] == binary and x['epoch'] == epoch for x in rows)
    captures = read(ROOT / 'accum_profiles.json')
    attempts = read(ROOT / 'accum_profile_attempts.json')
    profile_plan = read(ROOT / 'accum_profile_plan.json')
    assert len(captures) == 2 * len(profile_plan['pairs']) >= 8
    assert all(x['raw_token_bytes'] >= 4096 and x['epoch'] == epoch and
               x['benchmark_sha256'] == binary for x in captures)
    jobs = [json.loads(x) for x in (ROOT / 'jobs.jsonl').read_text().splitlines()]
    jobs = [x for x in jobs if x['benchmark_sha256'] == binary]
    started = [json.loads(x)['record'] for x in (ROOT / 'started.jsonl').read_text().splitlines()]
    started = [x for x in started if x['benchmark_sha256'] == binary]
    assert len(jobs) == len(started) == 35 + 23 * 8 + len(attempts)
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
    assert len(comparisons) == 237
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
    raw['journals/accum_controller_events.jsonl'] = (ROOT / 'accum_controller_events.jsonl').read_bytes()
    delivery = read(ROOT / 'profile_delivery_batch16.json')
    package = Path(delivery['path'])
    assert package.stat().st_size == delivery['bytes'] and sha(package.read_bytes()) == delivery['sha256']
    assert not DEST.exists()
    DEST.mkdir()
    names = ['session.json', 'authorization.json', 'initial_files_sha256.json', 'research_plan.json',
             'heldout_protocol.json', 'historical_exclusion_inventory.json', 'historical_final_attempts.json',
             'profile_delivery_batch16.json', 'environment_epoch.json', 'recovery_total.json',
             'compile/build17/source_sha256.json', 'compile/build17/completed.json',
             'readonly_reports/archive_accum_evidence.py',
             'readonly_reports/accum_actual_chain_count.json']
    names += [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob('accum_*')) if p.suffix in ['.json', '.py']]
    for folder in [ROOT / 'accum_profile_analysis', ROOT / 'drafts/conditional_accum_banks']:
        names += [str(p.relative_to(ROOT)) for p in sorted(folder.rglob('*')) if p.suffix in ['.json', '.py']]
    for name in sorted(set(names)):
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    with tarfile.open(package) as archive:
        for member in archive.getmembers():
            if Path(member.name).name in ['XCOMPUTE_README.md', 'profile_manifest.json', 'profile_manifest.csv']:
                path = DEST / 'profile_delivery' / Path(member.name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = archive.extractfile(member).read()
                path.write_bytes(data.replace(b'\r\n', b'\n') if path.suffix == '.csv' else data)
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
    pairs = {x['key']: x for x in read(ROOT / 'accum_discovery_plan.json')['discovery_pairs']}
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
                   qualification_jobs=35, three_round_pairs=23, full_comparisons=237,
                   full_elements=sum(int(x[1]) for x in comparisons), failed_elements=0,
                   paired_SQTT_captures=len(captures), profile_complete=True, selection=selection,
                   adopted=False, accepted_stable50=0, accepted_attributable30=0,
                   Goal_complete=False, GUI_verified=False)
    write(DEST / 'audit_summary.json', summary)
    disposition = '代表发现仍有确定回退，固定属性规则拒绝。' if not selection['eligible_for_confirmation'] else '代表发现门槛通过，但尚未完成独立确认，不构成采用。'
    (DEST / 'README.md').write_text('\n'.join([
        '# 固定 N128/K32/D2/W4：两套 FP32 累加缓冲', '',
        '同一K32块的两个K16部分分开累加到C0/C1，收尾逐元素相加。完整M所有权、A/B共享LDS、三套操作数缓冲、加载/读出/存储函数及selector不变；这属于既有H3的累加链因素，不是第四种所有权架构。未采用实验6035按源码/二进制SHA区分实现修订，6000/6001/6002/6003及1760个其他shader均保持。', '',
        f'benchmark SHA256：`{binary}`。35资格、23个三轮配对、237次全输出比较及真实guard通过；求和顺序改变，精度阈值未放宽。', '',
        '正式gfx936主循环的独立FP32 MMAC目标向量数，在M16/32/64分别从2/4/8变为4/8/16，静态MMAC总量不变；VGPR为67/99/163，6个双目标shader无spill/scratch。独立链增加是否改善实际重叠仍需SQTT和正式计时，不能单凭静态数量宣布收益。', '',
        f'16条实际代表GM（候选/当前CK）为 {selection["geometric_candidate_over_base"]:.6f}；三轮低CV且全部回退>2%的ID为 {selection["known_three_round_losses"]}。{disposition}', '',
        '属性域仍为96个原shape/53个原弱项。没有按赢家ID过滤，没有抽取新盲测seed，没有七block采用成绩；50/30目标计数仍为0/0，P6k和冻结BLAS不更新。', '',
        f'第16批交付{len(captures)}份有效SQTT，目标dispatch18/实际kernel/代码均核对；GUI未在本机运行。先读profile_delivery/XCOMPUTE_README.md，查看MMAC目标寄存器、DS read/等待重叠与尾部FP32相加。不同wave累计时钟不能相加当kernel墙钟。', '',
        '所有正式两臂均来自本轮同一冻结二进制、GPU4、epoch0。历史单累加6035跨构建值仅为诊断，不混入本轮成绩。paired_results_with_tflops.csv保留全部正向、回退、高CV、配置、实际实例、轮次、TFLOPS和R_frozen；原BLAS来自冻结GPU0 Native GroupGEMM。', '',
        '原1193shape和原Excel保持不变，只做本地实验提交，不push、不合入；用户修改保留。大型perf、二进制、厂商ISA和用户overlay源码不入Git，实际26份编译输入保留现场compile/build17。', '',
    ]))
    write(DEST / 'manifest.json', [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size,
                                      sha256=sha(p.read_bytes())) for p in sorted(DEST.rglob('*')) if p.is_file()])
    before = git('diff')
    assert not git('diff', '--cached', '--name-only').strip()
    index = ROOT / 'accum_archive.index'
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
    subprocess.run(['git', 'commit', '-m', 'P6l 双FP32累加缓冲23对与SQTT证据归档（实验用，不合入）'], cwd=REPO, env=env, check=True)
    for name, oid in owned.items():
        assert git('show', 'HEAD:' + name) == (REPO / name).read_bytes()
        subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + name], cwd=REPO, check=True)
    assert git('diff') == before and not git('diff', '--cached', '--name-only').strip()
    receipt = dict(at=r.stamp(), commit=git('rev-parse', 'HEAD').decode().strip(), files=len(owned),
                   raw_members=len(inventory), Git_blob_verified=True, user_worktree_preserved=True,
                   push=False, Goal_complete=False)
    write(ROOT / 'accum_archive_commit.json', receipt)
    print('NOSLC_RAW_AND_OWN_GIT_ARCHIVE_PASS', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
