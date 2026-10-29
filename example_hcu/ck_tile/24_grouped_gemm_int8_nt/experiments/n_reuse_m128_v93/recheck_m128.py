from pathlib import Path
import subprocess,os,json,resource,hashlib,csv
resource.setrlimit(resource.RLIMIT_CORE,(0,0));e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt')
assert not any(x in subprocess.check_output(['ps','-eo','args'],text=True) for x in ['aicc','probe.py','check_math.py','probe_balanced.py','probe_packed.py','validate_public_all'])
lib=e/'build/v89/libprobe.so';sha=hashlib.sha256(lib.read_bytes()).hexdigest();assert sha=='188bb1851578f3b939043e81032ccad73c7b688b18bf24d6146345d6a136e782'
math=json.loads((e/'results/d2l_v89_math.json').read_text());assert len(math)==248 and all(x['status']=='PASS' for x in math)
for p,h in json.loads((e/'build/public_v80/sources.json').read_text()).items():assert hashlib.sha256((e.parents[2]/p).read_bytes()).hexdigest()==h
rows=list(csv.DictReader((e/'docs/current_public_v80_1178.csv').open()));selected=[x for x in rows if int(x['M'])==128 and int(x['N'])%256==0 and int(x['K'])%64==0 and int(x['K'])<16384]
assert len(selected)==74 and len([x for x in selected if float(x['R'])<1])==9
out=e/'results/n_reuse_m128_v93';out.mkdir(exist_ok=True);assert not (out/'records.jsonl').exists()
(out/'manifest.json').write_text(json.dumps(dict(library_sha256=sha,library_source='v89 already full-range checked;no recompilation',gpu=7,scope='All74 native-eligible originalM128 shapes;four already-passing1232-tail shapes excluded;original NT,no scaling;4-wave Nreuse candidates versus same-library public6042;development screen only',selected_ids=[int(x['ID']) for x in selected],candidates=[6042,31001,31004,31011,31014]),indent=2)+'\n')
env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES='7',CUDA_VISIBLE_DEVICES='7');env['LD_LIBRARY_PATH']='/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:'+env.get('LD_LIBRARY_PATH','')
subprocess.run(['python3','-u',str(e/'probe.py'),','.join(x['ID'] for x in selected),str(lib),'n_reuse_m128_v93','6042,31001,31004,31011,31014'],env=env,check=True,timeout=1200)
print('V93_DONE',flush=True)
