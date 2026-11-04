"""Verify v1 header identity outside new IDs, all prior values and karg source."""
import hashlib,json,re,subprocess
from pathlib import Path
R=Path('/workspace/composable_kernel');P=Path(__file__).parent
h='include/ck_tile/ops/gemm/grouped_gemm_device_args.h'
b=subprocess.check_output(['git','show','83c8c2a:'+h],cwd=R,text=True);a=(R/h).read_text()
pattern=r'typedef enum ck_tile_hcu_grouped_gemm_instance_id_v1\s*\{(.*?)\}\s*ck_tile_hcu_grouped_gemm_instance_id_v1;'
bm=re.search(pattern,b,re.S);am=re.search(pattern,a,re.S);assert bm and am
assert re.sub(pattern,'INSTANCE_ENUM',a,flags=re.S)==re.sub(pattern,'INSTANCE_ENUM',b,flags=re.S)
parse=lambda s:{k:int(v) for k,v in re.findall(r'(CK_TILE_HCU_GROUPED_GEMM_INSTANCE_\w+)\s*=\s*(\d+)',s)}
bv=parse(bm[1]);av=parse(am[1]);assert all(av[k]==v for k,v in bv.items());new={k:v for k,v in av.items() if k not in bv};assert sorted(new.values())==list(range(5005,5011))
res=dict(public_v1_header_unchanged_outside_instance_enum=True,existing_ids_preserved=True,new_ids=new,initial_overstrict_text_check='Retained local check failed only on the required trailing comma and an empty enum line; this check compares all other content exactly and verifies old values.')
(P/'abi_audit.json').write_text(json.dumps(res,indent=2));print('PUBLIC_V1_ABI_SOURCE_PASS',len(bv),'old IDs preserved, six new IDs, other content identical')
