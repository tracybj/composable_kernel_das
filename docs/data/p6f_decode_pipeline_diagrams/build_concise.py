"""Architecture-only P6f decode illustration at one-pipeline scope; no GPU."""
import argparse
import json

from build_diagrams import Scene, P, ROOT


def box(s, x, y, w, h, lines, fill, size=27, color=None):
    s.rect(x,y,w,h,fill,radius=8)
    s.fit(x+16,y+11,w-32,h-22,lines,size,color,center=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--font',required=True)
    args=parser.parse_args()
    facts=json.loads((ROOT/'facts.json').read_text())
    assert facts['commit']=='c39e6df67de0a07d2f89ef224f421a99ed6ee9ca'
    s=Scene(1110,'BF16 NN Decode：共享LDS协作流水（P6f）',
            '以小M K16 / K32为例：wave按M/N负责不同输出块，各自累加完整K',args.font)
    s.section(195,'阶段一','主循环：CTA协作搬运，分组交错计算')
    s.fit(48,268,1704,35,['A、B共同进入四槽共享LDS，再读入三组操作数寄存器；每个stage同步后安全复用共享槽。'],28)
    s.card(48,326,250,146,'Global A / B',['CTA协作搬运','Direct-to-LDS'],P['gold'],26,True)
    s.rect(340,326,430,146,P['pink'])
    s.fit(356,345,398,40,['CTA共享LDS · 四槽'],29,center=True,bold_first=True)
    for j in range(4):
        s.rect(358+j*99,405,90,43,P['white'],P['line'],5)
        s.fit(362+j*99,415,82,27,[f'S{j} A/B'],22,center=True)
    s.card(812,326,350,146,'操作数寄存器',['R0 / R1 / R2','LDS读取后供MMAC使用'],P['blue'],26,True)
    s.rect(1204,326,548,146,P['purple'])
    s.fit(1222,347,512,43,['MMAC：计算当前K块'],32,P['white'],True,True)
    s.fit(1222,411,512,45,['FP32累加 → 各wave自己的C子块'],27,P['white'],True)
    for a,b in [(298,340),(770,812),(1162,1204)]:s.arrow([(a+6,403),(b-6,403)])
    for j,(wave,part) in enumerate([(0,'C00'),(1,'C01'),(2,'C10'),(3,'C11')]):
        x=48+j*435
        s.rect(x,505,399,45,P['pale'],radius=5)
        s.fit(x+12,516,375,29,[f'Wave {wave} → {part}（完整K）'],24,center=True)
    s.fit(48,584,175,57,['组0','Wave 0 / 1'],25,center=True,bold_first=True)
    s.fit(48,683,175,57,['组1','Wave 2 / 3'],25,center=True,bold_first=True)
    layout=[(240,260),(530,430),(990,260),(1280,472)]
    row0=[(['MMAC','计算当前块'],P['purple'],P['white']),
          (['LDS读后续操作数','预取后续A / B'],P['gold'],None),
          (['等待LDS','操作数就绪'],P['blue'],None),
          (['满足安全等待条件','CTA barrier保护共享槽'],P['red'],P['white'])]
    row1=[(['LDS读取','预取A / B'],P['gold'],None),
          (['等待LDS操作数就绪'],P['blue'],None),
          (['MMAC','计算当前块'],P['purple'],P['white']),
          (['满足安全等待条件','CTA barrier保护共享槽'],P['red'],P['white'])]
    for y,row in [(580,row0),(679,row1)]:
        for (x,w),(lines,fill,color) in zip(layout,row):box(s,x,y,w,88,lines,fill,25,color)
        for (x,w),(nx,_) in zip(layout,layout[1:]):s.arrow([(x+w+4,y+44),(nx-4,y+44)])
    s.fit(48,781,1704,32,['两组wave错开搬运与MMAC的次序，以尝试重叠等待和计算；方块宽度不表示实际耗时。'],25,P['muted'])
    s.section(837,'阶段二','K收尾与写回：完成计算，复用LDS排列输出')
    steps=[(48,280,['完成剩余K块','消化已预取数据'],P['purple'],P['white']),
           (365,330,['排空全局请求＋barrier','安全复用LDS'],P['red'],P['white']),
           (732,280,['FP32 → BF16','舍入转换'],P['blue'],None),
           (1049,330,['LDS排列＋读回','各wave独立输出片'],P['pink'],None),
           (1416,336,['Store C','16B向量写回'],P['green'],None)]
    for x,w,lines,fill,color in steps:box(s,x,918,w,99,lines,fill,27,color)
    for (x,w,*_), (nx,*_) in zip(steps,steps[1:]):s.arrow([(x+w+4,968),(nx-4,968)])
    s.fit(48,1044,1704,32,['K收尾：drain/steady停止无效预取，旧版仍可能冗余加载。C已累加完整K，尾部只排列写回。'],24)
    s.fit(48,1084,1704,24,['范围：P6f共享LDS的小M K16/K32；独立N尾块与各版本性能另见详细图。来源：冻结源码c39e6df。'],20,P['muted'])
    result=s.save('p6f_decode_concise')
    s.image.save(ROOT/'p6f_decode_concise.pdf','PDF',resolution=150)
    (ROOT/'concise_validation.json').write_text(json.dumps(dict(result='PASS',canvas=result,
        purpose='One-pipeline architecture communication; IDs, metrics, phase details moved to reference diagrams',
        source_commit=facts['commit'],gpu_jobs=0),indent=2))
    print('P6F_CONCISE_DIAGRAM_RENDER_PASS',json.dumps(result))


if __name__=='__main__':main()
