"""Archive derived facts and CK-owned analysis code; never archive vendor ISA."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
DOC=REPO/'docs/data/blas_pipeline_inventory_20261109'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary=json.loads((ROOT/'pipeline_summary.json').read_text())
    kernels=json.loads((ROOT/'pipeline_kernels.json').read_text())
    assert summary['gpu_jobs']==0 and len(kernels)==14
    protected=subprocess.check_output(['git','-C',str(REPO),'diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'])
    assert hashlib.md5(protected).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
    assert json.loads(Path('/workspace/v7_coverage/p6i_b_vector/stage_pause.json').read_text())['reason']=='USER_PAUSED'
    lines=['# BLASLt已选内核的流水结构清单', '',
           '范围限定为当前1193条BF16 NN GroupedGemm、gfx936的冻结历史选核记录：881条decode（M≤256）及312条prefill。不是BLASLt整个库、其他数据类型或其他转置布局的流水总数。', '',
           '本次仅CPU：对历史映射涉及的14个命名内核全部反汇编并核对稳态K循环，补齐此前只检查的6个内核。没有启动GPU、重新调优算法或复现性能，B加载实验与P6g复现均保持暂停。', '',
           '## 统计口径与结论', '',
           '- 命名内核实例：14个；其机器码SHA也为14个不同值。WGM、tile、wave映射或生成器的变体不能直接算成独立架构。',
           '- 数据通路大类：全部稳态K主循环都由两个输入descriptor执行Direct-to-LDS，随后从LDS读取操作数并MMAC，存在CTA barrier；归入CTA协作、共享LDS这一大类。',
           '- LDS缓冲组织：13个内核有四组循环LDS平面；1个内核有两个交替LDS平面。',
           '- 按K分块与LDS缓冲组织，归为以下3组主要配置。组内的operand布局、wave安排、等待位置仍有差异；这不是声称只有3份完全相同的流水实现。', '',
           '|主要配置|命名实例|decode shape|prefill shape|合计|',
           '|---|---:|---:|---:|---:|']
    for tag,label in [('k16_lds4','K16＋四槽共享LDS'),('k32_lds4','K32＋四槽共享LDS'),('k16_lds2','K16＋双缓冲共享LDS')]:
        x=summary['classes'][tag]
        lines.append(f'|{label}|{x["kernel_instances"]}|{x["decode"]}|{x["prefill"]}|{x["total"]}|')
    lines+=['|合计|14|881|312|1193|','',
            'decode中四槽两组共866/881=98.30%；其余15条为双缓冲组。此前709/881=80.5%是旧6个已反汇编内核的覆盖率，本次补齐其他8个后的统计范围不同，不能把它描述成新测了881条GPU性能。', '',
            '## 从指令验证，而非只从名称推断', '',
            '- K步长：全部14个内核的循环计数初始化都有对应K>>4或K>>5，与K16/K32名称一致。',
            '- 四槽：两个输入的LDS目标地址分别循环采用4组相同平面偏移。32 KiB实例为0/8192/16384/24576；64 KiB实例为0/16384/32768/49152。',
            '- 双缓冲：第12个内核的稳态目标与LDS读地址在0/16384两个平面交替；名称含PGR1，但不能因此叫一个LDS槽。其主循环采用vmcnt(0)、lgkmcnt(0)及CTA barrier，先读当前区、加载下一块到另一区后计算。',
            '- 四槽组主循环可见非零lgkmcnt等待和部分VMEM等待，结合多组operand寄存器与MMAC交错。这里记录的是静态控制流片段；有些片段包含不同wave组的备选路径，不能将静态barrier数直接当作每轮动态次数。',
            '- 四个LDS平面描述物理环，不等于整条流水只有四个在途或预读阶段；operand预装和剩余请求仍需单独计算。',
            '- 这14个已选内核的稳态路径中，没有发现A长期直驻寄存器、仅B使用wave私有LDS、零CTA barrier的主干。边界或epilogue的其他加载分支不据此单独算为新主干。', '',
            '## 几何配置', '',
            '以下M/N已从Native的C转置表示换回CK逻辑维度。几何有6种，不等于6种独立流水架构。', '',
            '|逻辑M×N×K|实例|decode|prefill|合计|', '|---|---:|---:|---:|---:|']
    for x in summary['logical_geometries']:
        tile='×'.join(map(str,x['logical_tile']))
        lines.append(f'|{tile}|{x["kernel_instances"]}|{x["decode"]}|{x["prefill"]}|{x["total"]}|')
    lines+=['', '## 全部实例', '',
            '|编号|逻辑tile|LDS平面|线程|LDS字节|VGPR|decode|prefill|合计|',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in kernels:
        tile='×'.join(map(str,x['logical_tile']))
        lines.append(f'|{x["index"]}|{tile}|{x["lds_buffers"]}|{x["threads"]}|{x["lds_bytes"]}|{x["vgpr"]}|{x["decode"]}|{x["prefill"]}|{x["total"]}|')
    lines+=['', '符号全名、shape ID清单、机器码SHA、主循环行号、LDS平面偏移与等待计数见pipeline_kernels.json/csv。原始vendor反汇编和code object仅保留在现场，不复制到CK仓库。', '',
            '## 实际命令与检查', '',
            '```bash',
            'python3 -B -u /workspace/v7_coverage/blas_pipeline_inventory_20261109/inventory.py',
            'python3 -B -u /workspace/v7_coverage/blas_pipeline_inventory_20261109/summarize.py',
            '```', '',
            '关键输出：CPU_INVENTORY_PASS kernels=14 shapes=1193 decode=881 prefill=312 GPU_jobs=0；PIPELINE_CLASSIFICATION_PASS。14个内核unknown_lines=0；原先6个已知内核的机器码SHA与P6d证据完全相同。所有计数与分组加和已断言。', '',
            '冻结BLAS code object SHA256：0100605e307ec45ab645712246805a04ccff6a402c2b86de77f6a3db88a52454。输入和工具依赖见dependencies.json。', '',
            '本目录脚本是现场审计副本。运行inventory.py需要先复制至CK仓库之外的工作目录并保持既有本地依赖；脚本会拒绝在CK仓库内写入vendor ISA。', '']
    DOC.mkdir(parents=True,exist_ok=False)
    text='\n'.join(lines)
    (ROOT/'report.md').write_text(text)
    (DOC/'README.md').write_text(text)
    for name in ['pipeline_summary.json','pipeline_kernels.json','pipeline_kernels.csv','dependencies.json','inventory.py','summarize.py','archive.py']:
        shutil.copyfile(ROOT/name,DOC/name)
    for p in DOC.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    validation=dict(ok=True,gpu_jobs=0,unknown_lines_total=0,all14_kernel_names_located=True,
                    preserved_previous_kernel_hashes=6,shape_counts=summary['total'],
                    groups_sum_to1193=True,classification_scope='steady K loop',
                    protected_user_diff_md5=hashlib.md5(protected).hexdigest(),p6i_state='paused',p6g_state='paused')
    (DOC/'validation.json').write_text(json.dumps(validation,indent=2))
    manifest={str(p.relative_to(DOC)):sha(p) for p in sorted(DOC.iterdir()) if p.is_file()}
    (DOC/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('CPU_PIPELINE_INVENTORY_ARCHIVE_PASS files='+str(len(manifest)),flush=True)


if __name__=='__main__':main()
