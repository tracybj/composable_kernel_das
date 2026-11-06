"""Official regressions, serial and with the same GPU4 gates as kernel tests."""
import runtime as r
import uuid
with r.locked():
 for name in ['selector_test','device_args_query_test','device_args_smoke']:
  r.execute(dict(key='regression_'+name+'_'+uuid.uuid4().hex[:10],kind='regression',regression_kind=name,env={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'0'},command=[str(r.REPO/'build/bin'/('tile_example_grouped_gemm_'+name))]))
print('OFFICIAL_REGRESSIONS_PASS')
