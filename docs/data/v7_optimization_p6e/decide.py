#!/usr/bin/env python3
"""Record research dispositions after every predeclared scope has finished."""
import collections,json,statistics
import analyze_all as analysis

p=analysis.P

def stats(rows):
    result=analysis.aggregate(rows)
    result['observational_labels']=dict(collections.Counter(row['observational_label'] for row in rows))
    if rows and all('after_vs_original_pct' in row for row in rows):
        result['median_delta_vs_original_pct']=statistics.median(row['after_vs_original_pct'] for row in rows)
        result['regress_vs_original_gt2']=sum(row['after_vs_original_pct']>2 for row in rows)
    return result

def main():
    expanded=json.loads((p/'small_expanded.json').read_text())
    assert len(expanded)==310,'Complete all scopes first'
    small=json.loads((p/'small_representatives.json').read_text())
    cache=[row for row in json.loads((p/'cache_expanded.json').read_text()) if row['policy_applies']]
    walk=json.loads((p/'cache_walk.json').read_text())
    groups={name:[row for row in expanded if row['strategy']==name] for name in ['k32_lowcta','drain_shortk','square_m128_256']}
    assert [len(groups[name]) for name in groups]==[21,189,100]
    decisions={
        'cache':dict(disposition='已验证，建议下一阶段以属性范围集成；本轮保持默认关闭',scope='gfx936/BF16/NN，common非ragged/nonempty，G<=1024，N%256=0，K>4096且K%16=0，并满足原tile上限及新增字节范围守卫；保留M01=8',metrics=stats(cache),evidence=['cache_expanded.json','cache_profiles.json','cache_isa.json','native_reference.json'],next='优先保留缓存提示隔离结果；1178同卡单launch仍比Native慢约2.8%，地址组织/推进节奏尚未验证，不能称全部达标。'),
        'cache_walk':dict(disposition='不采用M01=1，保留既有M01=8',metrics=stats(walk),evidence=['cache_walk.json'],next='去SLC后改回M01=1在三个代表长K对照均更慢；未继续重复遍历调参。'),
        'drain':dict(disposition='不整体提升为默认；保留独立实验及全部回退',scope='已完整验证M<=64、64<=K<2048、K%16=0、N%128=0、G<=1024，共189条',metrics=stats(groups['drain_shortk']),evidence=['small_expand_plan_drain_shortk.json','small_expanded.json','small_profiles.json'],next='先将稳态与尾段分开，减少每stage动态判断；当前76的VMEM/wave106.5→96.5，但分支103.5→258.5。控制开销解释尚未用独立实验量化，不直接删除LGKM0。'),
        'k32':dict(disposition='低CTA范围有额外收益，建议继续以属性门限验证集成；不全面开启',scope='gfx936/BF16/NN，common非ragged/nonempty，M<=64，128<=K<=4096，K%32=0，N%128=0，G<=1024，ceil(M/64)*ceil(N/128)*G<80，共21条',metrics=stats(groups['k32_lowcta']),evidence=['small_expand_plan_k32_lowcta.json','small_expanded.json','small_representatives.json','small_profiles.json'],dense_control=next({key:row[key] for key in ['id','M','N','K','G','delta_pct','observational_label']} for row in small if row['id']==358 and row['variant']=='k32'),next='48KB LDS使其LDS允许驻留从两CTA降到一CTA，低CTA收益与高CTA回退并存。21条中的高CV结果仍标为不确定；相对原5000的收益含此前5003的累计收益。K32+drain尚未组合验证。'),
        'k32n64':dict(disposition='不扩大、不采用全局N64策略；胜过旧5004不能等同胜过原路线',metrics=stats([row for row in small if row['variant']=='k32n64']),evidence=['small_representatives.json','small_profiles.json'],next='当前代表数据多数仍慢于N128或原5000；没有证据支持为扩大CTA而一概改N64。Native76/633虽是逻辑64x64x32，但不能只按tile照搬。'),
        'square':dict(disposition='扩大后以回退为主，不采用本版作为M128或M256通用策略；保留失败实验',scope='已完整验证128<=M<=256、64<=K<=4096、K%16=0、N%128=0、G<=1024，共100条，直接与原5000比较',metrics=stats(groups['square_m128_256']),by_m={str(m):stats([row for row in groups['square_m128_256'] if row['M']==m]) for m in [128,256]},evidence=['small_expand_plan_square_m128_256.json','small_expanded.json','scope_subgroups.json','small_profiles.json','square_diagnostic_profiles.json'],next='M128也不能一概提升：少数代表收益不能覆盖完整64条分层的多数回退。分层数据仅用于提出后续假设；所有100条回退计入，不逐shape挑最快。新增M128改善约12%、新增M256回退约5%，现有生产路径保持不变。square的A缓存提示和地址组织尚未独立验证。')}
    (p/'experiment_decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2))
    print('EXPERIMENT_DECISIONS_WRITTEN',len(decisions),'directions; no production-default changes')

if __name__=='__main__':main()
