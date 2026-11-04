"""Keep raw values intact; logical replacements are selected by protocol, never speed."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read():
 raw=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
 inv=ROOT/'invalidations.json';invalid=set(json.loads(inv.read_text()).get('raw_keys',[])) if inv.exists() else set()
 good={}
 for x in raw:
  if not x.get('ok') or x['key'] in invalid:continue
  key=x.get('logical_key',x['key']);row=dict(x);row.setdefault('logical_key',key);row.setdefault('environment_epoch',0);good[key]=row
 return raw,good

def epoch():
 p=ROOT/'environment_epoch.json';return json.loads(p.read_text())['id'] if p.exists() else 0
