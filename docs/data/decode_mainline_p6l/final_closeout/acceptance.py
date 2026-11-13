"""Seven-block numerical criteria; three-round discovery cannot count as success."""
import math,statistics

def classify(original,base,candidate):
    if len(base)!=7 or len(candidate)!=7:
        raise ValueError('Independent confirmation requires exactly seven paired blocks')
    for arm in [base,candidate]:
        for row in arm:
            if not all(math.isfinite(row[k]) for k in ['median_ms','cv']) or row['median_ms']<=0 or row['cv']<0:
                raise ValueError('Invalid timing or CV')
    ratios=[b['median_ms']/a['median_ms'] for a,b in zip(base,candidate)]
    geometric=math.exp(statistics.mean(math.log(x) for x in ratios))
    high_cv=any(x['cv']>.05 for x in base+candidate)
    improving_blocks=sum(x<.98 for x in ratios)
    stable=not high_cv and improving_blocks>=6 and geometric<=.95
    original_weak=bool(original and original.get('phase')=='decode' and 1<=original['ID']<=1193 and original['R_frozen']<.99)
    blas=original['frozen_BLAS_ms'] if original else None
    ra=[blas/x['median_ms'] for x in base] if blas else []
    rb=[blas/x['median_ms'] for x in candidate] if blas else []
    attributable=original_weak and stable and all(x<.99 for x in ra) and all(x>=.99 for x in rb)
    return dict(protocol='seven independent paired blocks; not discovery',base_ms=statistics.median(x['median_ms'] for x in base),candidate_ms=statistics.median(x['median_ms'] for x in candidate),paired_candidate_over_base=ratios,paired_geometric_candidate_over_base=geometric,blocks_improve_over_2pct=improving_blocks,high_cv=high_cv,stable_gain_at_least_5pct=stable,original_weak=original_weak,count_for_stable50=original_weak and stable,count_for_attributable30=attributable,base_R_frozen_blocks=ra,candidate_R_frozen_blocks=rb,blocks_regress_over_2pct=sum(x>1.02 for x in ratios),point_median_regression_over_2pct=statistics.median(x['median_ms'] for x in candidate)>1.02*statistics.median(x['median_ms'] for x in base))

def selftest():
    def arm(ms,cv=.01):return [dict(median_ms=x,cv=cv) for x in (ms if isinstance(ms,list) else [ms]*7)]
    old=dict(ID=805,phase='decode',R_frozen=.95,frozen_BLAS_ms=.95)
    yes=classify(old,arm(1),arm(.9));assert yes['count_for_stable50'] and yes['count_for_attributable30']
    drift=classify(old,arm(.9),arm(.8));assert drift['count_for_stable50'] and not drift['count_for_attributable30']
    cv=classify(old,arm(1),arm(.9,.051));assert cv['high_cv'] and not cv['count_for_stable50'] and not cv['count_for_attributable30']
    selected_best=classify(old,arm(1),arm([.1]+[1.1]*6));assert not selected_best['stable_gain_at_least_5pct']
    flip=classify(old,arm([.95]+[1]*6),arm(.85));assert flip['count_for_stable50'] and not flip['count_for_attributable30']
    historical_winner=classify(dict(old,R_frozen=1.01),arm(1),arm(.9));assert not historical_winner['count_for_stable50']
    synthetic=classify(None,arm(1),arm(.9));assert synthetic['stable_gain_at_least_5pct'] and not synthetic['count_for_stable50']
    for a,b in [(arm(1)[:3],arm(.9)[:3]),(arm(float('nan')),arm(.9))]:
        try:classify(old,a,b)
        except ValueError:pass
        else:raise AssertionError('Invalid confirmation input accepted')
    print('SEVEN_BLOCK_ACCEPTANCE_SELFTEST_PASS attribution_drift_CV_selection_synthetic_invalid_inputs')

if __name__=='__main__':selftest()
