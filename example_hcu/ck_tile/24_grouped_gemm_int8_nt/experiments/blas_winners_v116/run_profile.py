from pathlib import Path
import os,subprocess,json,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0));e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt');o=e/'results/blas_winners_v116';frozen=json.loads((e/'experiments/blas_winners_v116/winners.json').read_text())
assert all((o/('%04d_normal.json'%x['ID'])).exists() for x in frozen['results'])
assert not any(x in subprocess.check_output(['ps','-eo','args'],text=True) for x in ['aicc','check_math.py','screen.py','/blas_winners_v116/replay.py'])
env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES='7',CUDA_VISIBLE_DEVICES='7',DTK_VISIBLE_DEVICES='7');env['LD_LIBRARY_PATH']='/workspace/GroupGemm/baseline/hipblaslt-install/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:'+env.get('LD_LIBRARY_PATH','')
for i in [100,911,1126]:
 out=o/('profile_%04d'%i);out.mkdir(exist_ok=False);cmd=['/opt/dtk/bin/hipprof','--pmc','--pmc-type','3','-o',str(out/'capture'),'python3','-u',str(e/'experiments/blas_winners_v116/replay.py'),'--id',str(i),'--profile'];(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');print('PROFILE_START',i,flush=True)
 with (out/'run.log').open('w') as f:q=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 print('PROFILE_FINISH',i,q.returncode,[(p.name,p.stat().st_size) for p in out.iterdir() if p.is_file()],flush=True)
 assert q.returncode==0,(out/'run.log').read_text()[-2000:]
print('V116_PROFILE_DONE',flush=True)
