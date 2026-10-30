from pathlib import Path
import json,subprocess,collections,hashlib,struct
e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt');b=e/'build/blas_winners_v116';r=json.loads((b/'extracted.json').read_text());summ=[]
for x in r['kernels']:
 p=Path(x['object']);syms=subprocess.check_output(['readelf','-Ws',str(p)],text=True).splitlines();found=[l.split() for l in syms if l.split() and l.split()[-1]==x['kernel']];assert found and len({(l[1],l[2]) for l in found})==1,(x['algorithm_id'],found);line=found[0];start=int(line[1],16);size=int(line[2],0) if line[2].startswith('0x') else int(line[2])
 bounds='ELF function size'
 if size==0:
  next_symbols=[l.split() for l in syms if len(l.split())>=8 and l.split()[3]=='FUNC' and l.split()[4]=='GLOBAL' and l.split()[6]==line[6] and int(l.split()[1],16)>start];assert next_symbols
  stop=min(int(l[1],16) for l in next_symbols);size=stop-start;bounds='Zero-size ELF global FUNC;bounded by next global FUNC in same text section (includes final alignment padding)'
 assert size>1000,(x['algorithm_id'],line,size)
 q=subprocess.run(['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936','--start-address='+hex(start),'--stop-address='+hex(start+size),str(p)],capture_output=True,text=True);assert q.returncode==0,q.stderr;assert 'v_mmac_i32_' in q.stdout,(x['algorithm_id'],q.stdout[-500:]);out=b/str(x['algorithm_id']);(out/'full_kernel.isa').write_text(q.stdout)
 lines=q.stdout.splitlines();counts=collections.Counter();labels=[];markers=[]
 for i,l in enumerate(lines):
  if l.endswith('>:'):labels.append(l.strip())
  if '//' not in l:continue
  ss=l.split('//',1)[0].strip().split()
  if ss:counts[ss[0]]+=1
  if any(z in l for z in ['s_barrier','s_waitcnt','s_setprio','s_cmp_lt_u32','s_cmp_ge_u32','s_cmp_eq_u32','m0,','v_mmac_i32_']):markers.append(dict(line=i+1,instruction=l.strip()))
 row=dict(algorithm_id=x['algorithm_id'],symbol=x['kernel'],start=start,size=size,bounds=bounds,lines=len(lines),static_instruction_counts=dict(counts),labels=labels,markers=markers,isa_sha256=hashlib.sha256(q.stdout.encode()).hexdigest(),entry_only_lines=len((out/'kernel.isa').read_text().splitlines()),explanation='llvm-objdump --disassemble-symbols truncated at the next local label;an explicit address range is required. No evidence of a wrapper or separate compute-function call from this truncation.')
 (out/'full_summary.json').write_text(json.dumps(row,indent=2)+'\n');summ.append(row)
 print(json.dumps(dict(algorithm_id=x['algorithm_id'],size=size,lines=len(lines),labels=len(labels),key_instructions={k:v for k,v in counts.items() if k.startswith(('v_mmac','buffer_','ds_','s_barrier','s_waitcnt','s_setprio'))})),flush=True)
(b/'full_summary.json').write_text(json.dumps(summ,indent=2)+'\n')
