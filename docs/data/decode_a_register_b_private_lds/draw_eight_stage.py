"""Render the audited eight-slot A-register/B-private-LDS architecture."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / 'p6f_decode_pipeline_diagrams' / 'build_diagrams.py'
spec = importlib.util.spec_from_file_location('pipeline_scene_eight', TEMPLATE)
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
scene.ROOT = ROOT
Scene, P = scene.Scene, scene.P
NAME = 'decode_wave_eight_stage_prefetch'


def box(s, x, y, w, h, lines, fill, size=27, color=None, bold_first=False):
    s.rect(x, y, w, h, fill, radius=10)
    s.fit(x+16, y+15, w-32, h-30, lines, size, color,
          center=True, bold_first=bold_first)


def buffer_bank(s, y, title, prefix, fill):
    s.rect(340, y, 410, 170, fill)
    s.fit(356, y+15, 378, 40, [title], 27, center=True, bold_first=True)
    for j in range(8):
        x = 358+(j % 4)*95
        yy = y+64+(j // 4)*46
        s.rect(x, yy, 88, 38, P['white'], P['line'], 5)
        s.fit(x+6, yy+8, 76, 27, [prefix+str(j)], 24, center=True)


def source_facts():
    source = ROOT.parent/'v7_decode_wave_p6h/versions/trial03/source'
    pipeline = source/'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp'
    provider = source/'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_wave_device_args.cpp'
    code = pipeline.read_text()
    host = provider.read_text()
    for token in ['Bf16x4 a[Depth]', 'Depth * 1024u', 'k + (Depth + j) * 16',
                  'wait_global<2 * (Depth - 1)>', 'wait_global<2 * (Depth - 1 - j)>']:
        assert token in code, token
    assert 'k >= 128 && k % 128 == 0' in host
    assert 'launch_decode_nn<8>' in host and 'Depth * 4096' in host
    assert 's_barrier' not in code
    return dict(source_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [pipeline, provider]},
                k_per_stage=16, prefetch_depth=8, waves=4,
                cta_output_tile=[16, 128], wave_output_tile=[16, 32],
                wave_private_b_slot_bytes=1024, wave_private_b_bytes=8192,
                cta_b_lds_bytes=32768, a_register_banks=8,
                steady_refill_stage='t+8', slot='t%8', drain_stages=8,
                depth8_condition='K>=128 and K%128==0',
                cta_barriers=0, cross_wave_reduction=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', required=True)
    args = parser.parse_args()
    facts = source_facts()
    s = Scene(1220, 'BF16 NN Decode：Wave 独立八级预取流水',
              '示例：16×128输出块，4个wave沿N划分；K16分块、8级预取', args.font)
    s.section(195, '阶段一', '主循环：wave独立预取与计算，8槽循环复用')
    s.fit(48, 268, 1704, 35,
          ['A直接进入寄存器；B进入本wave私有LDS。先发出8个K块的加载，再逐块消费与补充。'], 28)

    box(s, 48, 326, 250, 170,
        ['Global A', 'A[M, K]', 'Direct-to-VGPR'], P['gold'], 27, bold_first=True)
    buffer_bank(s, 326, 'A寄存器 · 8组预取缓冲', 'A', P['blue'])
    s.arrow([(304, 411), (334, 411)])
    s.arrow([(756, 411), (1198, 411)])

    box(s, 48, 514, 250, 170,
        ['Global B', 'B[K, N]', 'Direct-to-LDS'], P['gold'], 27, bold_first=True)
    buffer_bank(s, 514, '本wave私有B-LDS · 8槽', 'S', P['pink'])
    box(s, 795, 514, 350, 170,
        ['B操作数寄存器', '读取当前LDS槽', '就绪后供MMAC使用'], P['blue'], 27, bold_first=True)
    s.arrow([(304, 599), (334, 599)])
    s.arrow([(756, 599), (789, 599)])
    s.arrow([(1151, 599), (1198, 599)])

    s.rect(1204, 326, 548, 358, P['purple'])
    s.fit(1224, 373, 508, 50, ['MMAC：计算当前K块 t'], 33, P['white'], True, True)
    s.fit(1224, 469, 508, 150,
          ['当前A(reg) × 当前B(reg)', 'FP32累加到当前wave的C子块', '独立累加完整K'], 29, P['white'], True)

    for j in range(4):
        x = 48+j*435
        s.rect(x, 714, 399, 47, P['pale'], radius=5)
        s.fit(x+12, 726, 375, 30,
              ['Wave '+str(j)+' → C'+str(j)+'（16×32）'], 25, center=True)
    s.fit(48, 778, 1704, 32,
          ['各wave负责不同N列；B私有LDS每wave为8 KiB，整个CTA共32 KiB。'], 25, P['muted'])

    flow = [
        (48, 350, ['预热', '发出第0～7块加载'], P['gold'], None),
        (436, 410, ['等待当前块 t 就绪', '读取B → MMAC计算'], P['purple'], P['white']),
        (884, 500, ['复用当前槽 t%8', '加载未来A/B块 t+8'], P['gold'], None),
        (1422, 330, ['推进到 t+1', '循环使用8个槽'], P['blue'], None),
    ]
    for x, w, lines, fill, color in flow:
        box(s, x, 832, w, 94, lines, fill, 27, color)
    for (x, w, *_), (nx, *_) in zip(flow, flow[1:]):
        s.arrow([(x+w+4, 879), (nx-4, 879)])
    s.fit(48, 946, 1704, 32,
          ['每消费1个K块，就补入后面第8块；各wave独立等待与推进，全程无需CTA barrier。'], 25, P['muted'])

    s.section(995, '阶段二', 'K收尾与写回：排空预取，复用私有LDS排列输出')
    tail = [
        (48, 280, ['排空末尾8个K块', '完成剩余MMAC'], P['purple'], P['white']),
        (365, 330, ['FP32 → BF16', '舍入转换'], P['blue'], None),
        (732, 280, ['写入私有LDS', '排列输出＋等待'], P['pink'], None),
        (1049, 330, ['读回16B向量', '等待LDS数据就绪'], P['blue'], None),
        (1416, 336, ['Store C', '按有效M行写回'], P['green'], None),
    ]
    for x, w, lines, fill, color in tail:
        box(s, x, 1076, w, 100, lines, fill, 27, color)
    for (x, w, *_), (nx, *_) in zip(tail, tail[1:]):
        s.arrow([(x+w+4, 1126), (nx-4, 1126)])
    s.fit(48, 1194, 1704, 26,
          ['收尾停止全局预取；每个C元素已累加完整K，尾部仅排列写回，无跨wave归约。'], 24)

    result = s.save(NAME)
    s.image.save(ROOT/(NAME+'.pdf'), 'PDF', resolution=150)
    ET.parse(ROOT/(NAME+'.svg'))
    texts = [x['text'] for x in s.labels]
    assert all('A'+str(j) in texts and 'S'+str(j) in texts for j in range(8))
    assert not any(token in text.lower() for token in
                   ['p6', 'git', 'commit', 'trial', '版本', '提交', '双缓冲'] for text in texts)
    validation = dict(result='PASS', canvas=result, architectural_phases=2, prefetch_depth=8,
                      gpu_jobs=0, checks=['All labels inside canvas', 'Eight A banks and eight B slots',
                          'Refill t+8 after consuming t', 'Eight-block drain without global prefetch',
                          'Wave-private LDS; full K per output; no CTA barrier or cross-wave reduction',
                          'SVG parses; no version/commit labels'], labels=texts, facts=facts)
    (ROOT/'eight_stage_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    print('EIGHT_STAGE_DECODE_DIAGRAM_PASS', json.dumps(result))


if __name__ == '__main__':
    main()
