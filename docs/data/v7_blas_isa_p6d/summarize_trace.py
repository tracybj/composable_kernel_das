#!/usr/bin/env python3
"""Audit SQTT capture without inventing an undocumented token decoder."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def main():
 jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
 rows=[]
 for r in jobs:
  if r['mode']!='trace' or not r.get('ok'):continue
  folder=Path(r['path'])/'profile';files=[]
  for p in sorted(folder.iterdir()):
   data=p.read_bytes();item=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   if p.suffix=='.out':
    words=data.decode('ascii').splitlines();assert all(re.fullmatch('[0-9a-fA-F]{4}',s) for s in words)
    item.update(format='hexadecimal 16-bit raw SQTT tokens, one word per line',words=len(words),raw_token_bytes=2*len(words),decoded_pc_or_latency=False)
   files.append(item)
  results=(folder/'results.txt').read_text();reported=int(re.search(r'SE\(0\) size\((\d+)\)',results)[1])
  assert sum(f.get('raw_token_bytes',0) for f in files)==reported
  rows.append(dict(key=r['key'],id=r['id'],backend=r['backend'],files=files,se0_reported_token_bytes=reported,scope='first active CU of first active SE (hardware output cu1/se0)',instruction_latency_analysis='not performed: installed Linux xprof emits raw token hex; supplied XCompute GUI is a Windows installer, no compatible decoder used',capture_timing_not_formal=True))
 result=dict(successful_traces=len(rows),traces=rows,decision='Probe succeeded. Do not collect more SQTT without a usable PC/latency decoder; use verified static ISA plus PMC. Raw capture is ready for XCompute GUI review.')
 (ROOT/'trace_summary.json').write_text(json.dumps(result,indent=2))
 print('SQTT_CAPTURE_PASS',len(rows),'raw bytes',sum(r['se0_reported_token_bytes'] for r in rows),'not decoded')

if __name__=='__main__':main()
