"""Commit verified factor source with an isolated index, preserving user edits."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
def git(*args,env=None,text=False):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=text)

def main():
 import time
 session=json.loads((ROOT/'session.json').read_text())
 assert session['gpu_deadline_epoch']<time.time()<session['deadline_epoch']
 assert not (ROOT/'phase_qualification.json').exists() and not (ROOT/'phase_discovery_results.json').exists()
 stages=json.loads((ROOT/'compile/build9/results.json').read_text());assert len(stages)==2 and all(x['rc']==0 for x in stages)
 built=json.loads((ROOT/'compile/build9/completed.json').read_text())
 audit=json.loads((ROOT/'phase_build_isa_audit.json').read_text());assert audit['ok'] and audit['old_shader_count']==1710
 assert built['benchmark_sha256']==audit['benchmark_sha256']==hashlib.sha256((REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark').read_bytes()).hexdigest()
 for name,sha in built['source_sha256'].items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==sha
 epoch=json.loads((ROOT/'environment_epoch.json').read_text())['id']
 stage=ROOT/'commit_preparation_phase';pure=stage/'own_tree';proof=json.loads((stage/'proof.json').read_text())
 for name in ['cpu_tests.json','host_syntax.json']:
  tests=json.loads((stage/name).read_text());assert tests and all(x['rc']==0 and x['gpu'] is False for x in tests)
 for x in proof:
  assert hashlib.sha256((REPO/x['path']).read_bytes()).hexdigest()==x['worktree_sha256']
  assert hashlib.sha256((pure/x['path']).read_bytes()).hexdigest()==x['pure_sha256']
 assert not git('diff','--cached','--name-only').strip()
 worktree={x['path']:(REPO/x['path']).read_bytes() for x in proof}
 index=stage/'isolated.index';assert not index.exists()
 env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
 subprocess.run(['git','read-tree','HEAD'],cwd=REPO,env=env,check=True)
 owned=[]
 for x in proof:
  n=x['path'];data=(pure/n).read_bytes();blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=REPO,input=data).decode().strip()
  subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+blob+','+n],cwd=REPO,env=env,check=True);owned.append((n,blob))
 changed=git('diff','--cached','--name-only',env=env,text=True).splitlines();assert changed and set(changed)<=set(worktree)
 subprocess.run(['git','diff','--cached','--check'],cwd=REPO,env=env,check=True)
 subprocess.run(['git','commit','-m','P6l提前计算MMAC原型：仅编译与静态验证，未做GPU验证（实验用，不合入）'],cwd=REPO,env=env,check=True)
 head=git('rev-parse','HEAD',text=True).strip()
 for name,blob in owned:subprocess.run(['git','update-index','--add','--cacheinfo','100644,'+blob+','+name],cwd=REPO,check=True)
 assert not git('diff','--cached','--name-only').strip()
 assert all((REPO/n).read_bytes()==data for n,data in worktree.items())
 def edits(patch):
  out={};current=None
  for line in patch.splitlines():
   if line.startswith('+++ b/'):current=line[6:];out.setdefault(current,[])
   elif current and line[:1] in ['+','-'] and not line.startswith(('+++','---')):
    if re.search(r'(GroupedGemmCandidateIdentity|CandidateExpected), \d+>',line):line=re.sub(r', \d+>',', COUNT>',line)
    out[current].append(line)
  return {k:v for k,v in out.items() if v}
 assert edits(git('diff',text=True))==edits((ROOT/'initial_tracked.patch').read_text())
 assert hashlib.md5(git('diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp')).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
 result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=head,changed_files=changed,qualification_count=0,discovery_count=0,GPU_validated=False,adopted=False,compiled_binary_sha256=built['benchmark_sha256'],old1710_machine_preserved=True,epoch=epoch,worktree_bytes_unchanged=True,initial_tracked_user_edits_preserved=True,readfirstlane_preserved=True,push=False)
 (ROOT/'phase_unvalidated_source_commit.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PHASE_COMPILED_ONLY_UNVALIDATED_SOURCE_COMMIT_PRESERVED_USER_EDITS',head,len(changed),flush=True)

if __name__=='__main__':main()
