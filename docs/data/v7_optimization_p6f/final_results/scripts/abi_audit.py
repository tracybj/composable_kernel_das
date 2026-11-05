"""Only appended instance enum values may change in the public v1 C header."""
import json,re,subprocess
import runtime as r
path='include/ck_tile/ops/gemm/grouped_gemm_device_args.h'
before=subprocess.check_output(['git','show',r.session()['head']+':'+path],cwd=r.REPO,text=True)
after=(r.REPO/path).read_text()
removed=re.findall(r'^\s*CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_\w+_V1 = 501[1-7],\s*$',after,re.M)
normalized=re.sub(r'^\s*CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_\w+_V1 = 501[1-7],\s*$','',after,flags=re.M)
tokens=lambda x:re.findall(r'\w+|[^\w\s]',x)
assert tokens(before)==tokens(normalized),'Public ABI header changed beyond new instance IDs'
record=dict(at=r.stamp(),public_header_only_appended_instance_ids=True,new_enum_declarations=[x.strip() for x in removed],before_head=r.session()['head'],working_header_sha256=r.sha(r.REPO/path),scope='All existing public tokens, structs, field layouts, declarations, enums/macros unchanged; new candidate IDs only')
(r.ROOT/'abi_audit.json').write_text(json.dumps(record,indent=2));print('PUBLIC_V1_ABI_TOKEN_PRESERVATION_PASS new_ids='+str(len(removed)))
