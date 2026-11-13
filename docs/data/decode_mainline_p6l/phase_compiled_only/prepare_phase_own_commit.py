"""Prepare an isolated source tree excluding every initial user edit.

This does not change the worktree, real index, HEAD, or stage any files.
"""
from pathlib import Path
import difflib, hashlib, json, re, subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')

def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO)

def main():
    session=json.loads((ROOT/'session.json').read_text())
    names=[x[6:] for x in git('diff','--name-only',session['head']).decode().splitlines() if False]
    patch=(ROOT/'initial_tracked.patch').read_text()
    names=re.findall(r'^\+\+\+ b/(.*)$',patch,re.M)
    stage=ROOT/'commit_preparation_phase'
    initial=stage/'initial';pure=stage/'own_tree'
    assert not stage.exists(),'Keep prior preparation immutable'
    initial.mkdir(parents=True)
    for n in names:
        p=initial/n;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(git('show',session['head']+':'+n))
    subprocess.run(['git','apply','--whitespace=nowarn',str(ROOT/'initial_tracked.patch')],cwd=initial,check=True)
    pins=json.loads((ROOT/'initial_files_sha256.json').read_text())
    proof=[]
    for n in names:
        assert hashlib.sha256((initial/n).read_bytes()).hexdigest()==pins[n]
        a=git('show',session['head']+':'+n).decode().splitlines(keepends=True)
        b=(initial/n).read_text().splitlines(keepends=True)
        c=(REPO/n).read_text().splitlines(keepends=True)
        # These two counts include both initial P6i IDs and our appended IDs.
        # Map their normalized initial line, then subtract exactly the two
        # initial IDs from the actual final count rather than discard ours.
        normal=c[:];special={}
        for i,line in enumerate(normal):
            if re.search(r'(?:GroupedGemmCandidateIdentity|CandidateExpected), \d+>',line):
                special[i]=line
                normal[i]=re.sub(r', \d+>',', 40>',line)
        mapping={}
        for op,bi,bj,ci,cj in difflib.SequenceMatcher(None,b,normal,autojunk=False).get_opcodes():
            if op=='equal':
                for j in range(bj-bi):mapping[bi+j]=ci+j
        replacements=[]
        for op,ai,aj,bi,bj in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if op=='equal':continue
            assert bi<bj,('Initial pure deletion needs explicit review',n,op,ai,aj,bi,bj)
            indices=[mapping[i] for i in range(bi,bj)]
            assert indices==list(range(indices[0],indices[0]+bj-bi)),(n,op,'initial edit split by own edit')
            lo,hi=indices[0],indices[-1]+1
            original=a[ai:aj]
            if lo in special:
                assert hi==lo+1 and len(original)==1
                original=[re.sub(r', (\d+)>',lambda m:', '+str(int(m[1])-2)+'>',special[lo])]
            else:assert c[lo:hi]==b[bi:bj],n
            replacements.append((lo,hi,original))
        d=c[:]
        for lo,hi,replacement in sorted(replacements,reverse=True):d[lo:hi]=replacement
        text=''.join(d)
        assert 'bvector' not in text and 'P6i:' not in text,n
        target=pure/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        proof.append(dict(path=n,initial_sha256=pins[n],worktree_sha256=hashlib.sha256((REPO/n).read_bytes()).hexdigest(),pure_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),initial_edits_removed=len(replacements),own_change=text!=''.join(a)))
    for n in ['include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_resource_phase.hpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_compact.hpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_wide.hpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative.hpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_shared_tuned.hpp',
      'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_cooperative_device_args.cpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_mainline.hpp',
      'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_shared.hpp',
      'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_mainline_device_args.cpp',
      'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_shared_device_args.cpp',
      'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_decode_mainline_selector_test.cpp']:
        assert n not in pins
        target=pure/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/n).read_bytes())
        proof.append(dict(path=n,new_own_file=True,worktree_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),pure_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),own_change=True))
    (stage/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('OWN_SOURCE_PREPARATION_PASS',len(proof),'ownchanged',sum(x['own_change'] for x in proof),flush=True)

if __name__=='__main__':main()
