"""Freeze property domains and unseen shape probes before any GPU measurements."""
import csv,json
from pathlib import Path
import runtime as r

ROOT=Path(__file__).resolve().parent
TABLE=r.REPO/'docs/data/v7_optimization_p6f/final_results/historical_envelope_incremental_1193.csv'

def load():
    rows=list(csv.DictReader(TABLE.open()))
    return {int(x['ID']):dict(id=int(x['ID']),**{k:int(x[k]) for k in ['M','N','K','G']},
            baseline=json.loads(x['enable_env']),expected=int(x['instance_id']),historical=x) for x in rows}

def short(x):
    return x['M']<=32 and x['N']<=4096 and x['N']%128==0 and 128<=x['K']<=1024 and x['K']%128==0 and x['G']<=64

def tiny(x):
    return x['M']==1 and x['N']<=4096 and x['N']%128==0 and 128<=x['K']<=2048 and x['K']%128==0 and x['G']<=8

def region(x,name):
    if name=='short_all':return short(x)
    if name=='short_low_cta':return short(x) and ((x['M']+15)//16)*(x['N']//128)*x['G']<=320
    if name=='short_narrow_n':return short(x) and x['N']<=512
    if name=='n256_k512_g64':return 1<=x['M']<=32 and x['N']==256 and x['K']==512 and x['G']==64
    if name=='tiny_low_group':return tiny(x)
    if name=='tiny_compact':return tiny(x) and x['N']<=2048 and x['K']>=512 and x['G']<=4
    raise ValueError(name)

REGIONS=['short_all','short_low_cta','short_narrow_n','n256_k512_g64','tiny_low_group','tiny_compact']

def main():
    cs=load();existing=[x for x in cs.values() if short(x) or tiny(x)]
    assert len(existing)==80
    tuples=[
      (1,128,128,1),(3,256,256,4),(7,384,512,16),(15,512,768,32),
      (17,256,1024,64),(31,512,512,64),(32,128,1024,16),(1,512,512,64),
      (8,256,512,16),(16,256,512,32),(32,256,512,32),(17,256,512,64),
      (3,256,512,64),(7,256,512,64),(15,256,512,64),(31,256,512,64),
      (1,1024,128,1),(1,1536,256,2),(1,2048,512,4),(1,3072,768,6),
      (1,4096,1024,8),(1,1536,1536,4),(1,2048,2048,8),(1,4096,2048,4),
    ]
    known={tuple(x[k] for k in ['M','N','K','G']) for x in cs.values()}
    probes=[]
    for i,t in enumerate(tuples):
        if t in known:
            t=(*t[:3],2)
        assert t not in known,t
        x=dict(id=-1001-i,**dict(zip(['M','N','K','G'],t)))
        assert short(x) or tiny(x)
        probes.append(x);known.add(t)
    boundaries=[cs[i] for i in [633,358,1185,1186]]
    predicates={
      'short_all':'M<=32; N<=4096 and N%128=0; 128<=K<=1024 and K%128=0; G<=64',
      'short_low_cta':'short_all and ceil(M/16)*(N/128)*G<=320',
      'short_narrow_n':'short_all and N<=512',
      'n256_k512_g64':'1<=M<=32; N=256; K=512; G=64',
      'tiny_low_group':'M=1; N<=4096 and N%128=0; 128<=K<=2048 and K%128=0; G<=8',
      'tiny_compact':'tiny_low_group and N<=2048 and K>=512 and G<=4',
    }
    data=dict(at=r.stamp(),task='Archived A-register/B-private-LDS Depth8 range validation',
      property_predicates=predicates,existing=existing,unseen_probes=probes,boundaries=boundaries,
      counts=dict(existing=80,unseen_probes=24,boundaries=4),
      old_training_ids=[76,681,1179,1180,1181,1182,1183,1184],
      first_pass_rounds=3,accuracy_repeats=3,formal_protocol='GPU4 serial; default selector; warmup10/samples30; one launch/sample; include_args_build0',
      novelty_baselines=['original V7/5000','small K16/5003','K16 drain/5007','K32/5008'],
      existing_baseline='Fresh run of each historical-envelope enabled configuration, no historical timing mixed into A/B',
      acceptance=dict(discovery='all member shapes have median change<-2%, all three paired rounds<-2%, no regression; median gain>=3%; at least four unseen probes',
          confirmation='Independent second complete three-round run of every member, same frozen predicate, no >2% regression, median gain>=3%; any high CV>5% remains uncertain rather than adopted',
          rollout='No production/default enable and no1193 table update in this validation phase'),
      table=str(TABLE),table_sha256=r.sha(TABLE),binary_sha256='fe5b5c7402683c0a6459beec4126ec8609ff0bb8bed8b1d114f0f5fdf44997a8',
      preserved_other_tasks=dict(p6i='paused',p6g='paused'))
    target=ROOT/'scope_plan.json'
    assert not target.exists()
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print('SCOPE_FROZEN existing=80 unseen=24 boundaries=4',flush=True)
    for name in REGIONS:
        print('REGION',name,'existing',sum(region(x,name) for x in existing),'unseen',sum(region(x,name) for x in probes))

if __name__=='__main__':main()
