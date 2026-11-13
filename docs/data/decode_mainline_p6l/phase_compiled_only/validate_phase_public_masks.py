"""All public-disable masks for new resource routes; pure CPU policy reader."""
from pathlib import Path
import datetime,hashlib,json,os,resource,subprocess
ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'cpu_phase_policy'
PREFIX='CK_TILE_GROUPED_GEMM_'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    prior=json.loads((FOLDER/'results.json').read_text());assert prior['ok'] and prior['case_count']==89493
    assert not any(x in prior['ldd'].lower() for x in ['libhsa','libamdhip','libgalaxyhip','libhipblas'])
    cases=[];lines=[]
    for disabled in range(64):
        for variant in [21,22,23,24]:
            for m in [1,16,17,32,33,64]:
                for k in [2048,6144]:
                    for family_disabled in [0,1]:
                        reference=None
                        for enable,fast_disabled in [('0','0'),('1','0'),('1','1')]:
                            env={PREFIX+'ENABLE_BF16_V7_NN':'1',
                                PREFIX+'ENABLE_BF16_NN_DECODE_MAINLINE':'1',
                                PREFIX+'DISABLE_BF16_NN_DECODE_MAINLINE':str(family_disabled),
                                PREFIX+'BF16_NN_DECODE_MAINLINE_VARIANT':str(variant),
                                PREFIX+'ENABLE_BF16_NN_DECODE_FAST_POLICY':enable,
                                PREFIX+'DISABLE_BF16_NN_DECODE_FAST_POLICY':fast_disabled}
                            index=len(cases);shape=dict(M=m,N=1024,K=k,G=8)
                            expected=None if family_disabled or (variant in [23,24] and m<17 and k<6144) or (variant==22 and m<17) else 6027 if variant==22 or (variant==23 and m>=17) else 6026
                            cases.append(dict(index=index,shape=shape,variant=variant,
                                public_disable_mask=disabled,family_disabled=family_disabled,
                                env=env,reference=reference,expected_new_id=expected))
                            values=[index,1,2,2,m,1024,k,8,0,0,0,disabled,0,len(env)]
                            lines.append(' '.join(map(str,values))+''.join(' '+name+' '+value for name,value in sorted(env.items())))
                            if reference is None:reference=index
    assert len(cases)==18432
    inputs=FOLDER/'public_disable_cases.json';assert not inputs.exists();inputs.write_text(json.dumps(cases,indent=2)+'\n')
    env={name:value for name,value in os.environ.items() if not name.startswith(PREFIX)}
    result=subprocess.run([str(FOLDER/'policy_equivalence')],input='\n'.join(lines)+'\n',capture_output=True,text=True,env=env,timeout=120)
    (FOLDER/'public_disable_run.log').write_text(result.stdout+result.stderr);assert result.returncode==0
    outputs={}
    for line in result.stdout.splitlines():
        values=list(map(int,line.split()));assert len(values)==7;outputs[values[0]]=values[1:]
    assert len(outputs)==len(cases)
    for case in cases:
        old,new=outputs[case['index']][:3],outputs[case['index']][3:]
        assert old[0]==new[0]==0
        if case['expected_new_id'] is not None:assert new[1]==case['expected_new_id'],(case,new)
        else:assert new==old,(case,old,new)
        if case['reference'] is not None:assert new==outputs[case['reference']][3:],(case,new)
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),ok=True,cpu_only=True,GPU=False,
        cases=len(cases),public_masks=64,variants=[21,22,23,24],family_disable_values=[0,1],
        geometries=12,fast_modes=[['0','0'],['1','0'],['1','1']],
        actual_extracted_source_sha256=prior['new_source_sha256'],
        executable_sha256=sha(FOLDER/'policy_equivalence'),inputs_sha256=sha(inputs),
        scope='Pure exact policy reader/selector; no ABI conversion, arch query, driver or GPU timing',
        script_sha256=sha(__file__))
    (FOLDER/'public_disable_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PHASE_PUBLIC_MASKS_CPU_PASS',len(cases),'NO_GPU',flush=True)

if __name__=='__main__':main()
