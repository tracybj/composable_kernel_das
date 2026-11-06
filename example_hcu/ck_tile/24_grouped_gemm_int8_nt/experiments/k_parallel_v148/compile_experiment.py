"""Build only the isolated K-parallel candidate;never changes production ABI."""
from pathlib import Path
import subprocess,json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);args=p.parse_args()
src=Path(__file__).resolve().parent;e=src.parents[1];root=e.parents[2];b=args.build.resolve();b.mkdir(parents=True,exist_ok=True)
identity={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in src.glob('*') if x.suffix in ['.hpp','.cpp']}
cmd=['/opt/dtk/bin/aicc','-xhip','--offload-arch=gfx936','-std=c++17','-O3','-shared','-fPIC','-DCK_ENABLE_INT8','-DCK_EXPERIMENTAL_BIT_INT_EXTENSION','-I'+str(root/'include'),'-I'+str(root/'library/include'),'-mllvm','-support-768-vgprs=true','-fno-offload-uniform-block','-mllvm','--lsr-drop-solution=1','-mllvm','-enable-post-misched=0','-mllvm','-amdgpu-early-inline-all=true','-fno-gpu-rdc',str(src/'bridge.cpp'),'-o',str(b/'libk_parallel.so')]
(b/'command.json').write_text(json.dumps(cmd,indent=2))
with (b/'compile.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
assert identity=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in src.glob('*') if x.suffix in ['.hpp','.cpp']}
(b/'identity.json').write_text(json.dumps(dict(source=identity,library_sha256=hashlib.sha256((b/'libk_parallel.so').read_bytes()).hexdigest()),indent=2))
