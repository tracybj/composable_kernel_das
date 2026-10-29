from pathlib import Path
import ctypes as C,os
class Weight(C.Structure):
 _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('data',C.c_void_p),('n',C.c_int32),('k',C.c_int32),('row_stride',C.c_int32),('reserved0',C.c_uint32),('reserved',C.c_uint64*2)]
class Group(C.Structure):
 _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('a',C.c_void_p),('c',C.c_void_p),('m',C.c_int32),('a_row_stride',C.c_int32),('c_row_stride',C.c_int32),('reserved0',C.c_uint32),('reserved',C.c_uint64*2)]
class WeightInfo(C.Structure):
 _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('group_count',C.c_int32),('n',C.c_int32),('k',C.c_int32),('format',C.c_int32),('packed_device_data',C.c_void_p),('packed_bytes',C.c_uint64),('reserved',C.c_uint64*2)]
class PlanInfo(C.Structure):
 _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('group_count',C.c_int32),('max_m',C.c_int32),('n',C.c_int32),('k',C.c_int32),('route',C.c_uint32),('legacy_instance',C.c_uint32),('packed_weight_bytes',C.c_uint64),('reserved',C.c_uint64*2)]
assert [C.sizeof(x) for x in [Weight,Group,WeightInfo,PlanInfo]]==[48,56,56,56]
def init(kind,**fields):
 x=kind();x.struct_size=C.sizeof(kind);x.abi_version=1
 for k,v in fields.items():setattr(x,k,v)
 return x
E=Path(__file__).resolve().parents[2]
libpath=Path(os.environ.get('CK_INT8_CACHED_BENCH_LIB',str(E/'build/cached_v97/libck_tile_int8_nt_cached_benchmark.so')))
lib=C.CDLL(str(libpath))
lib.ck_tile_int8_nt_cached_weights_create_v1.argtypes=[C.POINTER(Weight),C.c_int32,C.c_void_p,C.POINTER(C.c_void_p)]
lib.ck_tile_int8_nt_cached_weights_query_v1.argtypes=[C.c_void_p,C.POINTER(WeightInfo)]
lib.ck_tile_int8_nt_cached_weights_destroy_v1.argtypes=[C.c_void_p]
lib.ck_tile_int8_nt_cached_plan_create_v1.argtypes=[C.c_void_p,C.POINTER(Group),C.c_int32,C.POINTER(C.c_void_p)]
lib.ck_tile_int8_nt_cached_plan_query_v1.argtypes=[C.c_void_p,C.POINTER(PlanInfo)]
lib.ck_tile_int8_nt_cached_run_v1.argtypes=[C.c_void_p,C.c_void_p]
lib.ck_tile_int8_nt_cached_plan_destroy_v1.argtypes=[C.c_void_p]
lib.ck_tile_int8_nt_cached_last_error_v1.restype=C.c_char_p
lib.cached_i8_bench_create.argtypes=[C.c_void_p];lib.cached_i8_bench_create.restype=C.c_void_p
lib.cached_i8_bench_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p];lib.cached_i8_bench_run.restype=C.c_float
lib.cached_i8_bench_destroy.argtypes=[C.c_void_p];lib.cached_i8_bench_error.restype=C.c_char_p
def ok(status):
 assert status==0,(status,lib.ck_tile_int8_nt_cached_last_error_v1().decode())
def weight_info(handle):
 x=init(WeightInfo);ok(lib.ck_tile_int8_nt_cached_weights_query_v1(handle,C.byref(x)));return x
def plan_info(handle):
 x=init(PlanInfo);ok(lib.ck_tile_int8_nt_cached_plan_query_v1(handle,C.byref(x)));return x
