"""Archive existing evidence only; no GPU, compiler or experiment execution."""
from pathlib import Path
import datetime,gzip,hashlib,json,re,shutil,subprocess,time

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
BASE=REPO/'docs/data/decode_mainline_p6l'
PHASE=BASE/'phase_compiled_only'
FINAL=BASE/'final_closeout'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def copy(name,dest):
    p=ROOT/name;q=dest/name;q.parent.mkdir(parents=True,exist_ok=True)
    q.write_bytes(p.read_bytes());assert sha(q)==sha(p)
def inventory(folder):
    return [dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'manifest.json']
def edits(patch):
    out={};current=None
    for line in patch.splitlines():
        if line.startswith('+++ b/'):current=line[6:];out.setdefault(current,[])
        elif current and line[:1] in ['+','-'] and not line.startswith(('+++','---')):
            if re.search(r'(GroupedGemmCandidateIdentity|CandidateExpected), \d+>',line):
                line=re.sub(r', \d+>',', COUNT>',line)
            out[current].append(line)
    return {k:v for k,v in out.items() if v}

def main():
    session=read(ROOT/'session.json')
    assert session['gpu_deadline_epoch']<time.time()<session['deadline_epoch']
    assert read(ROOT/'gpu_stop.json')['reason']=='GPU_SUBMISSION_DEADLINE'
    audit=read(ROOT/'closeout/audit.json')
    assert audit['ok'] and audit['all_jobs_finished'] and audit['pending_started_jobs']==0
    assert not audit['performance_Goal_complete'] and audit['accepted_stable50']==audit['accepted_attributable30']==0
    assert not git('diff','--cached','--name-only').strip()
    original_diff=git('diff')
    assert edits(original_diff.decode())==edits((ROOT/'initial_tracked.patch').read_text())
    rf='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'
    assert hashlib.md5(git('diff','--',rf)).hexdigest()==session['protected_diff_md5']
    initial=read(ROOT/'initial_files_sha256.json')
    untracked=git('ls-files','--others','--exclude-standard','-z').decode().split('\0')
    untracked=[x for x in untracked if x]
    assert set(untracked)=={x[3:] for x in (ROOT/'initial_status.txt').read_text().splitlines() if x.startswith('?? ')}
    for n in untracked:assert sha(REPO/n)==initial[n]
    source_commit=read(ROOT/'phase_unvalidated_source_commit.json')
    assert source_commit['GPU_validated'] is False and source_commit['adopted'] is False
    assert git('rev-parse','HEAD').decode().strip()==source_commit['commit']
    pins=read(ROOT/'compile/build9/source_sha256.json')
    built=read(ROOT/'compile/build9/completed.json')
    assert pins==built['source_sha256'] and len(pins)==22
    assert sha(REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark')==built['benchmark_sha256']==audit['current_binary']
    # Preserve exact build input, including pre-existing user overlay, outside Git.
    snapshot=ROOT/'compiled_source_snapshot_phase'
    for n,s in pins.items():
        p=REPO/n;assert sha(p)==s
        q=snapshot/n;q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():assert sha(q)==s
        else:q.write_bytes(p.read_bytes())
    write(snapshot/'manifest.json',dict(at=stamp(),build='compile/build9',benchmark_sha256=built['benchmark_sha256'],source_sha256=pins,contains_initial_user_overlay=True,Git_committed=False))
    pure=read(ROOT/'commit_preparation_phase/proof.json')
    for x in pure:
        assert sha(REPO/x['path'])==x['worktree_sha256']
        blob=git('show','HEAD:'+x['path'])
        assert hashlib.sha256(blob).hexdigest()==x['pure_sha256']
    assert not PHASE.exists() and not FINAL.exists()
    PHASE.mkdir();FINAL.mkdir()
    phase_names=[
        'phase_hypothesis.json','phase_build_isa_audit.json','phase_unvalidated_source_commit.json',
        'phase_campaign_cpu_selector_audit.json','phase_baseline1193_cpu_audit.json',
        'phase_discovery_plan.json','phase_profile_rule.json',
        'compile/build9/results.json','compile/build9/completed.json','compile/build9/source_sha256.json',
        'compile/build9/0_command.json','compile/build9/1_command.json',
        'drafts/resource_interleave/schedule_hypothesis.json','drafts/resource_interleave/probe_result.json',
        'drafts/resource_interleave/probe_resources.json','drafts/resource_interleave/probe_schedule_audit.json',
        'drafts/resource_interleave/cpu_selector_tests.json','drafts/resource_interleave/audit_phase_probe.py',
        'drafts/resource_interleave/phase_probe.cpp','drafts/resource_interleave_cpu_model.py',
        'drafts/resource_interleave_cpu_model_result.json','cpu_phase_policy/results.json',
        'cpu_phase_policy/public_disable_equivalence.json','cpu_phase_policy/heldout_geometry_inventory.json',
        'cpu_phase_policy/ldd.txt','commit_preparation_phase/proof.json',
        'commit_preparation_phase/cpu_tests.json','commit_preparation_phase/host_syntax.json',
        'cpu_phase_policy_driver.py','validate_phase_public_masks.py','collect_resource_cpu_geometries.py',
        'phase_campaign.py','run_phase_discovery.py','analyze_phase_discovery.py',
        'audit_phase_qualification.py','audit_phase_build.py','phase_profiles_campaign.py',
        'run_final_confirmation_phase.py','build_phase_factor.py','prepare_phase_own_commit.py',
        'commit_phase_source.py','commit_phase_unvalidated.py',
    ]
    for n in phase_names:copy(n,PHASE)
    for n in ['compile/build9/0_build.log','compile/build9/1_build.log','drafts/resource_interleave/probe_build.log','cpu_phase_policy/run.log','cpu_phase_policy/public_disable_run.log']:
        p=ROOT/n;q=PHASE/(n+'.gz');q.parent.mkdir(parents=True,exist_ok=True)
        q.write_bytes(gzip.compress(p.read_bytes(),mtime=0));assert gzip.decompress(q.read_bytes())==p.read_bytes()
    outside=[]
    for n in ['cpu_phase_policy/cases.json','cpu_phase_policy/public_disable_cases.json','drafts/resource_interleave/phase_probe.o','compile/build9/reference_shaders.json']:
        p=ROOT/n;outside.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),Git_committed=False))
    write(PHASE/'outside_evidence.json',dict(compiled_sources_snapshot=str(snapshot),source_sha256=pins,large_artifacts=outside,benchmark_path=str(REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'),benchmark_sha256=built['benchmark_sha256']))
    write(PHASE/'UNRUN.json',dict(at=stamp(),GPU_qualification_jobs=0,planned_qualification_jobs=27,GPU_discovery_pairs=0,planned_discovery_pairs=46,SQTT_captures=0,profile_batch9_exists=False,full_adopted_domain_confirmation=False,performance_claim=False,adopted=False,reason='Original new-experiment deadline passed; postbuild GPU follow-up never started. Do not reset old session or bypass stop marker.',runner_execution_chain_verified=False))
    (PHASE/'README.md').write_text('''# 提前计算MMAC调度原型：仅编译和静态验证

**未做GPU query/smoke、精度、性能或SQTT；未采用，不计入50/30。** 6026/6027是H3协作LDS的一个调度因素，不是第四种所有权结构。新开关默认关闭，已有实例语义保留。

同样的五个几何比较两种指令次序后，CPU预登记规则选择所有wave先计算当前MMAC，再执行原有等待/barrier、读取未来操作数并预取。双目标探针均无spill/scratch；gfx936 VGPR总和493→467，wide M32/M64为83→73、128→112，compact仍63/86/133。三bank CPU模型和实际CFG审计只证明所检查的静态依赖，不能代替GPU正确性或时间收益。

六目标正式CPU构建于2026-11-12 17:57:39 UTC结束。当前benchmark SHA见compile/build9/completed.json；1710个既有shader机器码保持，10个新shader无spill/scratch，5个gfx936循环依赖通过只读检查。纯提交树2663个新selector、1063个Depth8及原回归通过；89493个policy场景/83521比较、18432个公开禁用位场景通过。所有测试和编译都在原20h截止前完成；截止后只读审计。

原1193配置的CPU选核未变，不是1193 GPU复测。源码自有部分本地提交5f4da600822a38fc2f8db8f0c13d230c9005d408，保留初始用户overlay；精确22份实际编译输入只在outside_evidence.json指定现场保存，未再次代为提交用户源码。旧compact收尾注释before computing与compute-first分支不符：实际最后MMAC先于ready(0)，排空后Store C；源码保持构建时字节，详见实际CFG审计。

phase_discovery_plan.json及计划脚本仅为未执行证据：27项资格、46配对和profile9均未运行。脚本链没有GPU端到端通过记录，不能宣称一键验证已跑通。原截止和gpu_stop仍有效，后续实验需用户另行授权的新时间窗口、重新检查入口并冻结输入；不能运行旧session绕过门禁。
''')
    write(PHASE/'manifest.json',inventory(PHASE))
    # Copy completed report and raw-log SHA index; never replace original measurements.
    for p in sorted((ROOT/'closeout').iterdir()):
        if p.is_file():copy(str(p.relative_to(ROOT)),FINAL)
    for n in ['session.json','gpu_stop.json','environment_epoch.json','recovery_total.json','recovery_067879389a154e40_resolved.json','h3_permission_recovery_audit.json','attempt1_closure.json','phase_unvalidated_source_commit.json','resource_source_commit.json','resource_archive_commit.json','h3_archive_commit.json','h3_factors_archive_commit.json','attempt1_archive_commit.json','diagnostic_commit.json','initial_files_sha256.json','public_abi_source_audit.json','closeout_audit.py','closeout_report.py','archive_final_closeout.py','acceptance.py','heldout_protocol.json','heldout_cpu_exclusion_audit.json','baseline_opportunity_map.json']:
        copy(n,FINAL)
    # Prove all existing archived file bytes match their committed Git blobs.
    tree={}
    for line in git('ls-tree','-r','HEAD','--',str(BASE.relative_to(REPO))).decode().splitlines():
        metadata,n=line.split('\t',1);mode,kind,oid=metadata.split();assert kind=='blob';tree[n]=oid
    existing=[]
    for n,oid in tree.items():
        data=(REPO/n).read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert actual==oid
        existing.append(dict(path=n,git_blob=oid,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    write(FINAL/'prior_archive_Git_blobs.json',existing)
    commits=[]
    for line in git('log','--format=%H %s',session['head']+'..HEAD').decode().splitlines():
        oid,title=line.split(' ',1);commits.append(dict(commit=oid,title=title,only_local=True,push=False))
    write(FINAL/'evidence_index.json',dict(at=stamp(),session=ROOT.as_posix(),original_T0=session['t0_utc'],new_experiment_cutoff_epoch=session['gpu_deadline_epoch'],hard_end_epoch=session['deadline_epoch'],immutable_inputs_verified=True,prior_archive_files_verified=len(existing),commits=commits,perf_packages=audit['perf_packages'],phase_archive=str(PHASE),phase_GPU_validated=False,Goal_achieved=False,accepted_stable50=0,accepted_cross30=0,adopted_families=[],raw_logs_all_verified='closeout/audit.json',raw_log_sha_index='closeout/raw_log_sha256.json'))
    write(FINAL/'protected_worktree_audit.json',dict(at=stamp(),tracked_initial_diff_preserved=True,untracked_initial_files_preserved=len(untracked),readfirstlane_diff_md5=session['protected_diff_md5'],source20_pure_Git_blobs_verified=True,actual_compiled22_source_sha_verified=True,current_unstaged_diff_sha256=hashlib.sha256(original_diff).hexdigest(),initial_patch_content_not_committed=True))
    (FINAL/'README.md').write_text('''# P6l 最终审计归档：目标未达成

本轮没有采用的新方案；稳定改善0/50、可归因跨桶0/30。完整报告见[closeout/README.md](closeout/README.md)，逐项状态见[closeout/requirements_audit.json](closeout/requirements_audit.json)。

408个三轮发现配对与166个七block原表确认均保留。后者GM candidate/base=1.053153，105条中位回退>2%，94条低CV且全部七block回退>2%，整域拒绝。28条个体稳定/14条个体跨桶不构成采用，不回填正式表。短线发现有负结果，最终确认未完成。最后提前计算MMAC原型只有编译/CPU/静态证据，见[phase_compiled_only](../phase_compiled_only/README.md)。

6632个已启动GPU作业全部终态，原始输出、门禁和版本逐项核对；全输出检查失败元素0，不代表性能验收通过。八批perf包见evidence_index.json及报告下载链接，保留Source/Wavefronts定位；本机未打开Windows XCompute。原P6k、冻结BLAS、1193 shape和旧Excel不变，projection文件明确是历史基线未替换，不是1193新鲜全量。

原新实验截止2026-11-12 22:53:23 UTC、硬截止2026-11-13 02:53:23 UTC保持。构建完成后的GPU接续没有启动，原型未验证；未归咎于等待用户或MKFD。恢复过一次是dmesg权限问题，按标准两次健康观察与固定探针处理，没有已确认的MKFD硬件事件。

evidence_index.json列出此前本地实验提交与8份perf包；prior_archive_Git_blobs.json证明原归档文件与当前Git对象一致。大型perf、二进制、厂商ISA及实际编译输入中的用户overlay留在现场，未新增到本归档。当前9处用户tracked及10份untracked内容保持，readfirstlane受保护，未push。原执行入口/控制器没有获得新时窗；未来实验必须另行授权，不能重置旧session。Goal仍未完成。
''')
    assert git('diff')==original_diff and not git('diff','--cached','--name-only').strip()
    write(FINAL/'manifest.json',inventory(FINAL))
    result=dict(at=stamp(),CPU_only=True,GPU_submitted=False,compiler_invoked=False,phase_files=len(inventory(PHASE)),final_files=len(inventory(FINAL)),prior_archive_files_verified=len(existing),tracked_and_untracked_user_content_preserved=True,Goal_complete=False)
    write(ROOT/'final_archive_preparation.json',result)
    print('FINAL_CLOSEOUT_ARCHIVE_PREPARED',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
