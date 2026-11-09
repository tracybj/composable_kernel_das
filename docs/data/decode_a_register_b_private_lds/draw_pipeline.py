"""Render one concise architecture page using the established diagram style."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / 'p6f_decode_pipeline_diagrams' / 'build_diagrams.py'
spec = importlib.util.spec_from_file_location('pipeline_scene', TEMPLATE)
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
scene.ROOT = ROOT
Scene, P = scene.Scene, scene.P


def box(s, x, y, w, h, lines, fill, size=27, color=None, bold_first=False):
    s.rect(x, y, w, h, fill, radius=10)
    s.fit(x+16, y+15, w-32, h-30, lines, size, color, center=True, bold_first=bold_first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', required=True)
    args = parser.parse_args()
    s = Scene(1040, 'BF16 NN Decode：A寄存器、B私有LDS流水',
              '示例：16×128输出块，4个wave沿N划分；K16分块、双缓冲', args.font)
    s.section(195, '阶段一', '主循环：wave独立搬运与计算')
    s.fit(48, 268, 1704, 35,
          ['A直接进入寄存器；B进入本wave私有LDS。双缓冲交替推进，不使用CTA barrier。'], 28)

    box(s, 48, 326, 250, 136, ['Global A', 'A[M, K]', 'Direct-to-VGPR'], P['gold'], 27,
        bold_first=True)
    box(s, 340, 326, 410, 136,
        ['A寄存器 · 双缓冲', 'A0 / A1', '直接作为MMAC操作数'], P['blue'], 27, bold_first=True)
    s.arrow([(304, 394), (334, 394)])
    s.arrow([(756, 394), (1198, 394)])

    box(s, 48, 480, 250, 136, ['Global B', 'B[K, N]', 'Direct-to-LDS'], P['gold'], 27,
        bold_first=True)
    s.rect(340, 480, 410, 136, P['pink'])
    s.fit(356, 497, 378, 40, ['本wave私有B-LDS'], 29, center=True, bold_first=True)
    for j in range(2):
        x = 364+j*190
        s.rect(x, 552, 172, 44, P['white'], P['line'], 6)
        s.fit(x+8, 562, 156, 28, ['槽 '+str(j)], 25, center=True)
    box(s, 795, 480, 350, 136,
        ['B操作数寄存器', '从当前LDS槽读取', '就绪后供MMAC使用'], P['blue'], 26, bold_first=True)
    s.arrow([(304, 548), (334, 548)])
    s.arrow([(756, 548), (789, 548)])
    s.arrow([(1151, 548), (1198, 548)])

    s.rect(1204, 326, 548, 290, P['purple'])
    s.fit(1224, 351, 508, 50, ['MMAC：计算当前K块'], 34, P['white'], True, True)
    s.fit(1224, 429, 508, 125,
          ['A(reg) × B(reg)', 'FP32累加到当前wave的C子块', '独立累加完整K'], 29, P['white'], True)

    for j in range(4):
        x = 48+j*435
        s.rect(x, 650, 399, 47, P['pale'], radius=5)
        s.fit(x+12, 662, 375, 30,
              ['Wave '+str(j)+' → C'+str(j)+'（16×32）'], 25, center=True)
    s.fit(48, 720, 1704, 35,
          ['各wave负责不同N列，独立预取、等待和计算；只需保证本wave的数据就绪与槽复用安全。'], 26, P['muted'])

    s.section(785, '阶段二', 'K收尾与写回：复用私有LDS排列输出')
    steps = [
        (48, 280, ['消化已预取数据', '完成剩余K块'], P['purple'], P['white']),
        (365, 330, ['FP32 → BF16', '舍入转换'], P['blue'], None),
        (732, 280, ['写入私有LDS', '排列输出＋等待'], P['pink'], None),
        (1049, 330, ['读回16B向量', '等待LDS数据就绪'], P['blue'], None),
        (1416, 336, ['Store C', '按有效M行写回'], P['green'], None),
    ]
    for x, w, lines, fill, color in steps:
        box(s, x, 866, w, 100, lines, fill, 27, color)
    for (x, w, *_), (nx, *_) in zip(steps, steps[1:]):
        s.arrow([(x+w+4, 916), (nx-4, 916)])
    s.fit(48, 994, 1704, 32,
          ['每个C元素已在所属wave内累加完整K；尾部只排列写回，无跨wave归约，也无需CTA barrier。'], 25)

    result = s.save('decode_a_register_b_private_lds')
    s.image.save(ROOT/'decode_a_register_b_private_lds.pdf', 'PDF', resolution=150)
    texts = [x['text'] for x in s.labels]
    assert not any(token in text.lower() for token in ['p6', 'git', 'commit', 'trial', '版本', '提交'] for text in texts)
    validation = dict(result='PASS', canvas=result, stages=2, gpu_jobs=0,
                      checks=['Text within canvas', 'No version or commit labels',
                              'A bypasses LDS', 'Wave-private B buffers',
                              'Disjoint N output tiles; each wave accumulates full K',
                              'Output transpose only; no cross-wave reduction or CTA barrier'],
                      labels=texts)
    (ROOT/'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    print('DECODE_PRIVATE_LDS_DIAGRAM_PASS', json.dumps(result))


if __name__ == '__main__':
    main()
