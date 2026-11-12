"""Promote audited CPU-built artifacts only after old GPU evidence is closed."""
from pathlib import Path
import fcntl,json,os,shutil,subprocess,time
import plan,runtime as r
ROOT=Path(__file__).resolve().parent
ISO=ROOT/'isolated_resource_build'

def main():
    r.no_core()
    with (ROOT/'final_controller.lock').open('a') as controller,r.locked():
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
        assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json','user_pause.json'])
        assert r.read(ROOT/'profile_delivery_batch7.json')['captures']==8
        closure=r.read(ROOT/'attempt1_closure.json');assert closure['original_main_complete']==166 and closure['whole_family_rejected']
        mechanism=r.read(ROOT/'resource_mechanism_decision.json');assert mechanism['proceed'] and mechanism['diagnostic_only']
        assert mechanism['batch7_analysis_sha256']==plan.sha(ROOT/'h3_regression_profile_analysis/profile_analysis.json')
        state=r.read(ISO/'followup_registry_completion/status.json');assert state['state']=='compiled_cpu_only'
        assert all(x['rc']==0 for x in r.read(ISO/'followup_registry_completion/results.json'))
        isa=r.read(ISO/'resource_full_isa_audit.json');assert isa['ok'] and isa['old_shader_count']==1700
        revision=r.read(ISO/'revisions/registry_completion/revision.json')
        pins=revision['isolated_source_sha256']
        assert all(plan.sha(ISO/'source'/name)==sha for name,sha in pins.items())
        old=r.read(ROOT/'frozen_plan.json');assert old['phase']=='h3_regression_diagnosis'
        assert plan.sha(plan.BENCH)==old['benchmark_sha256']=='f8facea69107f345acfc98ce3e6e6deb1cca41ac803e2731bc0ab925e6736bfb'
        assert all(plan.sha(plan.REPO/name)==sha for name,sha in old['worktree_source_sha256'].items())
        patch=ISO/'resource_integration_complete.patch';assert plan.sha(patch)==revision['complete_patch_sha256']
        subprocess.run(['git','apply','--check','--whitespace=error-all',str(patch)],cwd=plan.REPO,check=True)
        destination=ROOT/'compile/build8';assert not destination.exists();destination.mkdir(parents=True)
        backup=destination/'previous';backup.mkdir()
        binaries=['tile_example_grouped_gemm_device_args_benchmark','tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke','tile_example_grouped_gemm_selector_test','tile_example_grouped_gemm_decode_depth8_selector_test','tile_example_grouped_gemm_decode_mainline_selector_test']
        provenance=dict(at=r.stamp(),complete=False,source_sha256=pins,old_benchmark_sha256=old['benchmark_sha256'],
            new_benchmark_sha256=state['benchmark_sha256'],isolated_source=str(ISO/'source'),isolated_build=str(ISO/'build'),
            revision_sha256=plan.sha(ISO/'revisions/registry_completion/revision.json'),patch_sha256=plan.sha(patch),
            shader_audit_sha256=plan.sha(ISO/'resource_full_isa_audit.json'),binary_artifacts=[],GPU=False,adopted=False)
        r.write(destination/'promotion_started.json',provenance)
        shutil.copy2(ROOT/'frozen_plan.json',backup/'frozen_plan.json')
        for name in pins:
            source=plan.REPO/name
            if source.exists():target=backup/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for name in binaries:
            source=ISO/'build/bin'/name;actual=plan.REPO/'build/bin'/name
            old_sha=plan.sha(actual);shutil.copy2(actual,backup/name)
            ldd=subprocess.run(['ldd',str(source)],capture_output=True,text=True,check=True)
            assert 'not found' not in ldd.stdout and str(ISO) not in ldd.stdout
            provenance['binary_artifacts'].append(dict(name=name,old_sha256=old_sha,new_sha256=plan.sha(source),ldd=ldd.stdout))
        subprocess.run(['git','apply','--whitespace=error-all',str(patch)],cwd=plan.REPO,check=True)
        assert all(plan.sha(plan.REPO/name)==sha for name,sha in pins.items())
        for item in provenance['binary_artifacts']:
            target=plan.REPO/'build/bin'/item['name'];temporary=target.with_name(target.name+'.p6l-resource.tmp')
            assert not temporary.exists();shutil.copy2(ISO/'build/bin'/item['name'],temporary)
            assert plan.sha(temporary)==item['new_sha256'];os.replace(temporary,target)
        assert plan.sha(plan.BENCH)==state['benchmark_sha256']
        provenance.update(complete=True,completed_at=r.stamp(),original_user_overlay_preserved=True,
            note='CPU-only artifact/source promotion. Formal path, exact isolated source and external dependencies; not GPU accuracy or adoption evidence.')
        r.write(ROOT/'resource_promotion.json',provenance);r.write(destination/'source_sha256.json',pins)
        r.write(destination/'results.json',r.read(ISO/'followup_registry_completion/results.json'))
        r.write(ROOT/'active_build.json',dict(folder='compile/build8',at=r.stamp(),isolated_build_provenance='resource_promotion.json',CPU_only=True))
        print('RESOURCE_PROMOTION_COMPLETE_NO_GPU',state['benchmark_sha256'],flush=True)

if __name__=='__main__':main()
