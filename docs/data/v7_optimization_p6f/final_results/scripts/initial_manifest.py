"""Hash the small run inputs and frozen measurement/configuration artifacts."""
import json,subprocess
from pathlib import Path
import runtime as r
paths=[r.ROOT/'session.json',Path('/workspace/CLAUDE.md'),Path('/workspace/v7_coverage/prompts/P6f_tail_cache_decode_16h.md'),Path('/workspace/shapes_1178.csv'),Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv'),Path('/workspace/bf16NN-groupgemm.csv'),Path('/workspace/composable_kernel/docs/data/v7_best_known/best_current_1193.csv'),Path('/workspace/v7_coverage/p6f_review/review.json'),Path('/workspace/v7_coverage/p6f_review/assessment_facts.json'),Path('/workspace/Grouped_Geem/blaslt_tuning_tools/results/bf16_nn_compare_20261027_191315/groupgemm/run_config.json'),Path('/workspace/v7_coverage/p6c/new_blas/run_config.json'),Path(r.BLAS),Path('/workspace/Grouped_Geem/blaslt_tuning_tools/build/libgroupgemm_tune.so'),Path('/workspace/v7_coverage/p6e/bench_cache')]
rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=r.sha(p)) for p in paths]
out=dict(at=r.stamp(),head=r.session()['head'],inputs=rows,protected_diff_sha256=r.sha(r.ROOT/'protected_user.patch'),git_status_initial=' M include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp',task_prompt_executed=True,no_full_gpu_suite=True)
(r.ROOT/'initial_manifest.json').write_text(json.dumps(out,indent=2));print('INITIAL_MANIFEST',len(rows))
