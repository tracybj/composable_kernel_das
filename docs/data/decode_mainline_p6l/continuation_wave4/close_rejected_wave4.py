from pathlib import Path
import copy,csv,json,shutil,tarfile
import runtime as r,plan,wave4_profiles_campaign as original
ROOT=Path(__file__).resolve().parent
RULE=ROOT/'wave4_profile_rule_partial.json'
REGPLAN=ROOT/'wave4_profile_partial_plan.json'
CAPTURES=ROOT/'wave4_profiles.json'
ATTEMPTS=ROOT/'wave4_profile_attempts.json'
ANALYSIS=ROOT/'wave4_profile_analysis'
def deliver():
    captures=r.read(CAPTURES);assert len(captures)==2*len(r.read(REGPLAN)['pairs'])
    codes={(x['ID'],x['arm']):x for x in r.read(ANALYSIS/'profile_isa.json')}
    metrics={(x['ID'],x['arm']):x for x in r.read(ANALYSIS/'profile_analysis.json')['rows']}
    target=ROOT/'delivery/batch14_wave4_PARTIAL';assert not target.exists();target.mkdir(parents=True)
    manifest=[];text=['# PARTIAL：N128 四 wave，仅两组完整 SQTT','',
        '27对正式三轮发现已完整结束，候选整域拒绝。本包仅含原定8对中的2对完整SQTT（620与805，共4份），其余采集未完成。ID686在CU0/1/2/3/7仅有稀疏trace，CU15的profiler中应用中止；原因未充分定位。原始失败均保留，未降低4096字节门槛，不继续搜索CU。本包不能称为完整第14批或全部预登记profile完成。',
        '', '完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。',
        '先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。',
        '重点比较每CTA聚合SALU/VALU/LDS指令与实际耗时，核对减少B重复LDS消费是否抵消寄存器上升及wave并行度变化；不要只比每wave计数。直接两臂全CTA MMAC、A生产者/global字节与输出工作不变，B消费理论减半、A消费不变，实际计数及HBM仍须核对。主循环K32D2/三操作数组、vmcnt(0)/LG0及barrier安全依赖保持，LDS为M32的20KiB或M64的24KiB。平均驻留本身不证明理论上限；不同wave累计Latency不能相加当kernel墙钟。',
        '本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。','']
    for cap in captures:
        code=codes[cap['ID'],cap['arm']];metric=metrics[cap['ID'],cap['arm']]
        assert code['unknown_lines']==0 and cap['raw_token_bytes']>=4096
        source=Path(next(x['path'] for x in cap['files'] if x['path'].endswith('.perf'))).parent
        relative=Path('captures')/source.name;shutil.copytree(source,target/relative)
        entry={k:v for k,v in cap.items() if k not in ['files','raw_log']};entry['files']=[]
        entry.update(kernel_name=code['captured_kernel'],machine_sha256=code['sha256'],phases=metric['geometry'])
        for item in cap['files']:
            p=target/relative/Path(item['path']).relative_to(source)
            assert p.stat().st_size==item['bytes'] and plan.sha(p)==item['sha256']
            entry['files'].append(dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=plan.sha(p)))
        manifest.append(entry);perf=next(x for x in entry['files'] if x['path'].endswith('.perf'))
        text += [f'## ID{cap["ID"]} {cap["pair"]} {cap["role"]}','',
            f'- MNKG {cap["shape"]}；实例{cap["actual_instance"]}；GPU4/epoch{cap["epoch"]}。',
            f'- 文件 `{perf["path"]}`；dispatch18/XCD0/active-CU-index{cap["active_CU_index"]}。',
            f'- kernel `{code["captured_kernel"]}`。',f'- 资源 {metric["geometry"]}。',
            f'- 正式candidate/base变化{cap["point_delta_pct"]:+.3f}%；平均驻留wave/CU {metric.get("mean_resident_waves_per_CU")}；HBM读取 {metric.get("hbm_read_bytes")}字节；wait/wavecycle {metric.get("wait_any_over_wave_cycles")}。',
            f'- 主循环PC `{code["main_loop_candidates"]}`。','']
        text += ['  指令PC：`'+z['text']+'`' for z in code['instruction_anchors'][:16]]
    r.write(target/'profile_manifest.json',manifest)
    fields=['ID','M','N','K','G','pair','comparison_kind','role','actual_instance','physical_gpu','epoch','selected_dispatch','active_CU_index','raw_token_bytes','perf_path','perf_sha256']
    with (target/'profile_manifest.csv').open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fields);writer.writeheader()
        for item in manifest:
            perf=next(x for x in item['files'] if x['path'].endswith('.perf'))
            row={k:item.get(k) for k in fields};row.update(item['shape']);row.update(perf_path=perf['path'],perf_sha256=perf['sha256']);writer.writerow(row)
    for p in [RULE,REGPLAN,ATTEMPTS,ANALYSIS/'profile_analysis.json',ANALYSIS/'h3_static_schedule.json',ANALYSIS/'same_candidate_code_across_G.json']:
        shutil.copy2(p,target/p.name)
    (target/'XCOMPUTE_README.md').write_text('\n'.join(text)+'\n')
    package=target.parent/'P6l_batch14_wave4_PARTIAL_SQTT.tar.gz';assert not package.exists()
    with tarfile.open(package,'w:gz') as archive:archive.add(target,arcname=target.name)
    r.write(ROOT/'profile_delivery_batch14.json',dict(at=r.stamp(),path=str(package),bytes=package.stat().st_size,sha256=plan.sha(package),captures=len(captures),gui_verified=False,local_integrity_checked=True))
    print('REGRESSION_PROFILE_DELIVERY_PASS',package,package.stat().st_size,plan.sha(package),flush=True)

def main():
 original.REGPLAN=REGPLAN;original.RULE=RULE
 original.analyze();deliver()
 receipt=r.read(ROOT/'profile_delivery_batch14.json');receipt.update(partial=True,expected_capture_pairs=8,accepted_capture_pairs=2,failed_profile_jobs=1)
 r.write(ROOT/'profile_delivery_batch14.json',receipt)
 rows=r.read(ROOT/'wave4_discovery_results.json');assert len(rows)==27
 selected=[x for x in rows if not x['control']];assert len(selected)==16
 losses=[x['ID'] for x in selected if not x['high_cv'] and all(b['median_ms']/a['median_ms']>1.02 for a,b in zip(x['base_rounds'],x['candidate_rounds']))]
 assert losses
 r.write(ROOT/'wave4_rejected_closed.json',dict(at=r.stamp(),state='rejected',formal_discovery_complete=True,qualification_jobs=35,discovery_pairs=27,known_losses=losses,profile_complete=False,valid_profile_pairs=2,original_profile_pairs=8,retained_failed_profile_job='e0_h3wave4_id686_base_cu15_bd384a9a_5095deded4',benchmark_sha256=plan.sha(plan.BENCH),epoch=r.read(ROOT/'environment_epoch.json')['id'],adopted=False,Goal_complete=False,new_heldout_seed_drawn=False))
 print('REJECTED_WAVE4_CLOSED_WITH_EXPLICIT_PARTIAL_PROFILES',flush=True)
if __name__=='__main__':main()
