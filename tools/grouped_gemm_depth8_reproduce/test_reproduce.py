"""CPU/fake-process tests for correctness gates and recovery/continuation safety."""
import contextlib
import copy
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import health
import plan
import report
import reproduce
import runtime as r

SOURCE=plan.REPO/'docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv'


def example(kind='accuracy',selected=5000,repeats=1):
    lines=['CONFIG arch=gfx936 dtype=bf16 layout=NN groups=4 m=1 n=2048 k=768 variable_axis=none '+
           ('timing=off output_redzone_elements=16' if kind=='accuracy' else 'timing=on output_redzone_elements=0')+
           ' warmup=10 samples=30 launches_per_sample=1 include_args_build=0 num_cu=0 blas_streams=1',
           f'SELECT instance_id={selected}']
    for i in range(repeats):
        lines.append(f'ACCURACY backend=ck repeat={i} checked=257 failed=0 max_abs=0')
        for name,count in [('ck_guard',0),('blas_guard',0),('ck_redzone',32),('blas_redzone',32),('ck_vs_blas',8192)]:
            lines.append(f'ACCURACY backend={name} repeat={i} checked={count} failed=0 max_abs=0')
    if kind=='perf': lines.append('PERF backend=ck_device_args median_ms=0.1 min_ms=0.09 max_ms=0.11 cv=0.01')
    return '\n'.join(lines+['SUMMARY result=PASS'])


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        r.write(self.root/'session.json',dict(t0_epoch=time.time(),t0_monotonic=0,deadline_epoch=time.time()+36000,physical_gpu=4))
        self.case=plan.load_table(SOURCE)['cases'][0]
        self.manifest=dict(plan_sha256='fixture',benchmark_sha256='fixture',blas_sha256='fixture',probe={},
                           table=dict(cases=[self.case]))
        r.write(self.root/'frozen_plan.json',self.manifest);self.site=r.Site(self.root)

    def test_environment_does_not_inherit_previous_recipe(self):
        inherited={plan.V7:'0',plan.V7+'_N_TAIL':'2','CK_TILE_GROUPED_GEMM_UNKNOWN':'bad','PATH':'/bin'}
        e=plan.environment({plan.V7:'1'},inherited)
        self.assertEqual({k:v for k,v in e.items() if k.startswith(plan.PREFIX)},{plan.V7:'1'})
        self.assertEqual(e['HIP_VISIBLE_DEVICES'],e['CUDA_VISIBLE_DEVICES']);self.assertEqual(e['OMP_NUM_THREADS'],'4')
        with self.assertRaises(ValueError):plan.environment({plan.V7+'_UNKNOWN':'1'})
        with self.assertRaises(ValueError):plan.environment({plan.V7:'yes'})

    def test_depth8_overlay_changes_only_six_recipes_and_keeps_history(self):
        source=plan.load_table(SOURCE)
        modified=plan.with_depth8_overlay(source)
        hits=[x for x in modified['cases'] if x['candidate_overlay_applied']]
        self.assertEqual([x['ID'] for x in hits],list(range(1179,1185)))
        self.assertTrue(all(x['expected']==6003 and x['env'][plan.DEPTH8]=='1' for x in hits))
        self.assertTrue(all(x['accuracy_repeats']==3 for x in hits))
        self.assertFalse(any(plan.DEPTH8 in x['env'] for x in source['cases']))
        for old,new in zip(source['cases'],modified['cases']):
            self.assertEqual(old['historical_CK_ms'],new['historical_CK_ms'])
            self.assertEqual(old['frozen_BLAS_ms'],new['frozen_BLAS_ms'])
            if new['ID'] not in range(1179,1185):self.assertEqual(old,new)

    def test_original_and_overlay_rows_share_a_csv_schema(self):
        rows=plan.with_depth8_overlay(plan.load_table(SOURCE))['cases']
        with (self.root/'recipes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (self.root/'recipes.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),1193)

    def test_depth8_flag_is_cleared_and_disable_wins_in_probe(self):
        inherited={plan.DEPTH8:'1',plan.V7:'1','PATH':'/bin'}
        self.assertNotIn(plan.DEPTH8,plan.environment({plan.V7:'1'},inherited))
        case=copy.deepcopy(self.case)
        case['env'].update({plan.DEPTH8:'1',plan.PREFIX+'DISABLE_BF16_NN_DECODE_DEPTH8':'1'})
        self.assertEqual(plan.policy_fields(case)[-1],0)
        case['env'][plan.PREFIX+'DISABLE_BF16_NN_DECODE_DEPTH8']='0'
        self.assertEqual(plan.policy_fields(case)[-1],1)

    def test_gpu_deadline_is_two_hours_before_total_and_never_reset(self):
        self.site.session=dict(t0_epoch=100,deadline_epoch=43300,gpu_deadline_epoch=36100)
        with patch.object(r.time,'time',return_value=36050):
            self.assertEqual(self.site.gpu_remaining(),50)
            with self.assertRaises(r.QueueStopped):self.site.gate('late')
        self.assertEqual(r.read(self.site.stop)['reason'],'GPU_SUBMISSION_DEADLINE')

    def test_unreadable_kernel_log_stops_before_any_gpu(self):
        with patch.object(r.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='Operation not permitted')):
            with self.assertRaises(r.QueueStopped):self.site.kernel_events('unreadable')
        self.assertEqual(r.read(self.site.stop)['reason'],'KERNEL_LOG_UNREADABLE')

    def test_qualification_jobs_are_never_accepted_as_full_replay_rows(self):
        for kind,rd in [('accuracy',0),('perf',1),('perf',2),('perf',3)]:
            r.append(self.root/'jobs.jsonl',dict(kind=kind,round=rd,ok=True,purpose='qualification',ID=self.case['ID']))
        self.assertEqual(report.completed(self.root,self.manifest),({},[]))

    def test_actual_recipe_dimensions_and_commands(self):
        data=plan.load_table(SOURCE)
        self.assertEqual(len(data['cases']),1193)
        self.assertEqual(sum(x['phase']=='decode' for x in data['cases']),881)
        self.assertEqual([x['ID'] for x in data['cases'][-15:]],list(range(1179,1194)))
        for kind in ['accuracy','perf']:
            cmd=plan.command(self.case,kind)
            self.assertEqual(cmd[0],str(plan.BENCH));self.assertNotIn('--repeat=20',cmd)
            self.assertIn('--launches-per-sample=1',cmd)
            self.assertFalse(any('--instance' in x for x in cmd))

    def test_full_output_and_real_redzones_required(self):
        r.parse_output(example(),'accuracy',5000)
        for text in [example().replace('ck_vs_blas','cpu_sample'),
                     example().replace('backend=ck_vs_blas repeat=0 checked=8192 failed=0 max_abs=0',
                                       'backend=ck_vs_blas repeat=0 checked=8192 failed=0 max_abs=1e309'),
                     example().replace('ck_redzone','no_redzone'),
                     example().replace('checked=32 failed=0','checked=0 failed=0'),
                     example().replace('backend=blas_redzone repeat=0 checked=32 failed=0','backend=blas_redzone repeat=0 checked=32 failed=1')]:
            with self.assertRaises(ValueError):r.parse_output(text,'accuracy',5000)

    def test_instance_error_mkfd_rc0_and_bad_timing_rejected(self):
        with self.assertRaises(ValueError):r.parse_output(example(),'accuracy',5016)
        with self.assertRaises(ValueError):r.parse_output(example()+'\nMKFD','accuracy',5000)
        r.parse_output(example('perf'),'perf',5000)
        for old,new in [('median_ms=0.1','median_ms=nan'),('launches_per_sample=1','launches_per_sample=20'),
                        ('samples=30','samples=20'),('output_redzone_elements=0','output_redzone_elements=16')]:
            with self.assertRaises(ValueError):r.parse_output(example('perf').replace(old,new),'perf',5000)

    def test_frozen_binary_change_stops(self):
        p=self.root/'binary';p.write_bytes(b'changed')
        with self.assertRaises(r.QueueStopped):self.site.identity(p,'not-the-same')
        self.assertEqual(r.read(self.site.stop)['reason'],'BINARY_OR_DEPENDENCY_CHANGED')

    def check_fixture(self,stdout,rc=0,memory=70*1024*1024):
        original=Path.read_text
        def read(path,*a,**kw):
            return f'MemAvailable: {memory} kB\n' if str(path)=='/proc/meminfo' else original(path,*a,**kw)
        with patch.object(r.subprocess,'run',return_value=SimpleNamespace(stdout=stdout,stderr='',returncode=rc)),patch.object(Path,'read_text',read):
            return self.site.check('fixture')

    def test_fresh_health_requires_target_clean_rc_and_idle(self):
        row='4 50C 80W auto 1000W 0% 0.0% 0.0% 0.0% Normal\n'
        self.assertTrue(self.check_fixture(row)[0])
        self.assertFalse(self.check_fixture(row+'MKFD')[0])
        self.assertFalse(self.check_fixture(row,rc=1)[0])
        self.assertFalse(self.check_fixture(row.replace('4 ','0 ',1))[0])
        with self.assertRaises(r.QueueStopped):self.check_fixture(row.replace('0.0%','1.0%',1))

    def test_low_memory_stops_without_gpu(self):
        with self.assertRaises(r.QueueStopped):self.check_fixture('4 50C 80W auto 1000W 0% 0.0% 0 0 N',memory=1)
        self.assertEqual(r.read(self.site.stop)['reason'],'LOW_HOST_MEMORY')

    def test_service_event_before_job_marks_environment(self):
        with patch.object(self.site,'kernel_events',return_value=['hymgr unregister']),patch.object(self.site,'check') as check:
            with self.assertRaises(r.EnvironmentFailure):self.site.gate('fixture')
            check.assert_not_called()
        self.assertTrue(self.site.mark.exists())

    def test_observation_timeout_leaves_process_and_stops_submissions(self):
        folder=self.root/'timeout';folder.mkdir()
        process=SimpleNamespace(pid=999999,wait=Mock(side_effect=subprocess.TimeoutExpired('fixture',1)),kill=Mock(),terminate=Mock())
        with patch.object(r.subprocess,'Popen',return_value=process):
            rc,pid=self.site.run_program(['fixture'],folder,{},1)
        self.assertIsNone(rc);self.assertEqual(pid,999999)
        process.kill.assert_not_called();process.terminate.assert_not_called()
        self.assertTrue(self.site.stop.exists())

    def test_rc0_service_event_excludes_actual_fake_child(self):
        fixture=Path(__file__).with_name('fake_benchmark.py')
        spec=dict(logical_key='fake_service',kind='accuracy',ID=self.case['ID'],shape={k:self.case[k] for k in ['M','N','K','G']},
                  expected=self.case['expected'],env=self.case['env'],attempt='incident',round=0,
                  command=[sys.executable,str(fixture),'--case-json',json.dumps(self.case),'--kind','accuracy'])
        with patch.object(self.site,'gate',return_value=dict(at=r.stamp())),patch.object(self.site,'identity'),\
             patch.object(self.site,'kernel_events',return_value=['hymgr unregister']):
            with self.assertRaises(r.EnvironmentFailure):self.site.execute(spec)
        last=report.records(self.root)[-1]
        self.assertEqual(last['rc'],0);self.assertFalse(last['ok']);self.assertTrue(self.site.mark.exists())

    def test_dry_run_has_no_gpu_or_health_commands(self):
        shadow=self.root/'bin';shadow.mkdir();marker=self.root/'forbidden_gpu_command'
        tool=shadow/'hy-smi';tool.write_text('#!/bin/sh\ntouch '+str(marker)+'\nexit 73\n');tool.chmod(0o755)
        env=dict(os.environ,PATH=str(shadow)+':'+os.environ['PATH'])
        out=self.root/'dry'
        result=subprocess.run([sys.executable,str(Path(reproduce.__file__)),'--plan',str(SOURCE),
                               '--output',str(out),'--dry-run'],env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse(marker.exists());self.assertFalse((out/'jobs.jsonl').exists())
        self.assertIn('DRY_RUN_PASS',result.stdout)

    def recovery_fixture(self,healthy,transitions,rc=0):
        r.write(self.site.mark,dict(at=datetime.datetime.fromtimestamp(1000,datetime.timezone.utc).isoformat(),epoch=0))
        clock=[1000.0]
        def sleep(seconds):clock[0]+=seconds
        def check(key,attempt):return healthy[attempt-1],dict(at=datetime.datetime.fromtimestamp(clock[0],datetime.timezone.utc).isoformat())
        def program(cmd,folder,env,timeout):
            (folder/'run.log').write_text(example(repeats=3));return rc,123
        with patch.object(r,'locked',contextlib.nullcontext),patch.object(health,'probe_evidence',return_value={}),\
             patch.object(health.plan,'sha',return_value=r.PROBE_SHA),patch.object(health.time,'time',lambda:clock[0]),\
             patch.object(health.time,'monotonic',lambda:clock[0]),patch.object(health.time,'sleep',side_effect=sleep) as sleeps,\
             patch.object(self.site,'check',side_effect=check) as checks,patch.object(self.site,'kernel_events',side_effect=transitions),\
             patch.object(self.site,'run_program',side_effect=program) as probe:
            try:health.recover(self.site)
            except (r.QueueStopped,ValueError):pass
            return sleeps.call_args_list,checks.call_count,probe.call_count

    def test_recovery_three_observations_spaced_and_fresh_probe(self):
        sleeps,checks,probes=self.recovery_fixture([True,True,True],[['transition'],[],[],[]])
        self.assertEqual(checks,3);self.assertEqual(probes,1)
        self.assertEqual([x.args[0] for x in sleeps],[60,60]);self.assertEqual(self.site.epoch(),1)
        self.assertFalse(self.site.mark.exists());self.assertFalse(self.site.stop.exists())

    def test_recovery_max3_and_probe_failure_are_terminal(self):
        _,checks,probes=self.recovery_fixture([False,True,False],[[],[],[]])
        self.assertEqual(checks,3);self.assertEqual(probes,0);self.assertTrue(self.site.stop.exists())

    def test_known_correct_probe_failure_stops(self):
        _,checks,probes=self.recovery_fixture([True,True],[[],[],[]],rc=1)
        self.assertEqual(checks,2);self.assertEqual(probes,1);self.assertTrue(self.site.stop.exists())
        self.assertEqual(self.site.epoch(),0)

    def test_recovery_90minute_limit_blocks_probe(self):
        r.write(self.root/'recovery_total.json',dict(seconds=5400,attempts=1))
        _,checks,probes=self.recovery_fixture([True,True],[[],[],[]])
        self.assertEqual((checks,probes),(0,0));self.assertTrue(self.site.stop.exists())

    def test_real_cpu_child_interruption_resumes_whole_shape(self):
        for suffix in reproduce.REGRESSIONS:
            r.append(self.root/'jobs.jsonl',dict(kind='regression',ok=True,logical_key='regression_'+suffix,plan_sha256='fixture'))
        fixture=Path(__file__).with_name('fake_benchmark.py')
        def command(case,kind):return [sys.executable,str(fixture),'--case-json',json.dumps(case),'--kind',kind]
        original=self.site.execute
        def interrupted(spec):
            if spec['kind']=='perf' and spec['round']==2:raise RuntimeError('CPU controller interrupted at a safe boundary')
            return original(spec)
        with patch.object(r,'locked',contextlib.nullcontext),patch.object(self.site,'gate',side_effect=lambda key:dict(at=r.stamp())),\
             patch.object(self.site,'identity'),patch.object(self.site,'kernel_events',return_value=[]),\
             patch.object(reproduce,'PILOT',[self.case['ID']]),patch.object(plan,'command',side_effect=command):
            with patch.object(self.site,'execute',side_effect=interrupted):
                with self.assertRaises(RuntimeError):reproduce.run_once(self.site)
            self.assertEqual(len(report.completed(self.root,self.manifest)[0]),0)
            reproduce.run_once(self.site)
            done,partial=report.completed(self.root,self.manifest)
            self.assertEqual(len(done),1);self.assertEqual(len(partial),1)
            n=len(report.records(self.root));reproduce.run_once(self.site)
            self.assertEqual(len(report.records(self.root)),n)
            self.assertEqual(len([x for x in report.records(self.root) if x.get('kind')=='accuracy']),2)
            other=self.root/'cross_epoch';other.mkdir()
            chosen=done[self.case['ID']]['source_keys']
            selected=[copy.deepcopy(x) for x in report.records(self.root) if x.get('key') in chosen]
            for x in selected:
                if x['kind']=='perf' and x['round']==3:x['epoch']+=1
                r.append(other/'jobs.jsonl',x)
            self.assertFalse(report.completed(other,self.manifest)[0])


if __name__=='__main__':unittest.main(verbosity=2)
