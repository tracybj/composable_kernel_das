// SPDX-License-Identifier: MIT
#ifndef CK_TILE_INT8_NT_CACHED_H
#define CK_TILE_INT8_NT_CACHED_H
#include <stddef.h>
#include <stdint.h>
#if defined(__GNUC__)
#define CK_TILE_INT8_CACHED_API __attribute__((visibility("default")))
#else
#define CK_TILE_INT8_CACHED_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Additive INT8-only API. Existing grouped-GEMM ABI structs and FP16/BF16
// entries are neither extended nor replaced. All strides count elements.
enum ck_tile_int8_nt_cached_status_v1 {
 CK_TILE_INT8_CACHED_SUCCESS_V1=0,
 CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1=-1,
 CK_TILE_INT8_CACHED_BAD_VERSION_V1=-2,
 CK_TILE_INT8_CACHED_BAD_STRUCT_SIZE_V1=-3,
 CK_TILE_INT8_CACHED_UNSUPPORTED_ARCH_V1=-4,
 CK_TILE_INT8_CACHED_UNSUPPORTED_V1=-5,
 CK_TILE_INT8_CACHED_BUSY_V1=-7,
 CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1=-100
};
enum ck_tile_int8_nt_cached_route_v1 {
 CK_TILE_INT8_CACHED_LEGACY_RAW_V1=0,
 CK_TILE_INT8_CACHED_LINEAR_RAW_V1=1,
 CK_TILE_INT8_CACHED_LINEAR_RAW_RECORD1024_V1=2,
 CK_TILE_INT8_CACHED_LINEAR_TRANSPOSED_RAW_V1=3,
 CK_TILE_INT8_CACHED_LINEAR_TRANSPOSED_RAW_RECORD1024_V1=4,
 CK_TILE_INT8_CACHED_PACKED_WEIGHT_V1=5,
 CK_TILE_INT8_CACHED_PACKED_M128_N256_A2_V1=6,
 CK_TILE_INT8_CACHED_PACKED_M128_N128_A4_V1=7,
 CK_TILE_INT8_CACHED_PACKED_M64_N256_A2_V1=8,
 CK_TILE_INT8_CACHED_PACKED_M64_N64_A2_V1=9
};
typedef struct ck_tile_int8_nt_cached_weights_impl* ck_tile_int8_nt_cached_weights_handle_v1;
typedef struct ck_tile_int8_nt_cached_plan_impl* ck_tile_int8_nt_cached_plan_handle_v1;

typedef struct ck_tile_int8_nt_cached_weight_desc_v1 {
 uint32_t struct_size,abi_version;
 const int8_t* data; // Device NT weight [N,K],16-byte aligned.
 int32_t n,k,row_stride;
 uint32_t reserved0;
 uint64_t reserved[2];
} ck_tile_int8_nt_cached_weight_desc_v1;
typedef struct ck_tile_int8_nt_cached_group_desc_v1 {
 uint32_t struct_size,abi_version;
 const int8_t* a; // Device [M,K].
 int32_t* c;     // Device [M,N]. Signed INT32 output,no scale.
 int32_t m,a_row_stride,c_row_stride;
 uint32_t reserved0;
 uint64_t reserved[2];
} ck_tile_int8_nt_cached_group_desc_v1;
typedef struct ck_tile_int8_nt_cached_weights_info_v1 {
 uint32_t struct_size,abi_version;
 int32_t group_count,n,k,format;
 const int8_t* packed_device_data; // Read-only;valid until weights_destroy.
 uint64_t packed_bytes;
 uint64_t reserved[2];
} ck_tile_int8_nt_cached_weights_info_v1;
typedef struct ck_tile_int8_nt_cached_plan_info_v1 {
 uint32_t struct_size,abi_version;
 int32_t group_count,max_m,n,k;
 uint32_t route,legacy_instance;
 uint64_t packed_weight_bytes;
 uint64_t reserved[2];
} ck_tile_int8_nt_cached_plan_info_v1;

// Every v1 struct must set struct_size=sizeof(struct),abi_version=1 and zero
// reserved fields. Weights share N/K;M may differ across groups,including0.
// Prepare is asynchronous on stream. Source NT weights remain valid and
// immutable until handle destruction:raw fallback borrows them. Updating
// weights requires a new prepare;there is no implicit pointer-keyed cache.
// The packed format,when supported,is[G,K/64,N/16,4,16,16]. It copies INT8
// bytes without quantization;FP16/BF16 tensors are never accepted by this API.
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_create_v1(
 const ck_tile_int8_nt_cached_weight_desc_v1* weights,int32_t group_count,
 void* stream,ck_tile_int8_nt_cached_weights_handle_v1* result);
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_query_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 weights,ck_tile_int8_nt_cached_weights_info_v1* info);
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_destroy_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 weights);

// Create binds A/C addresses and uploads argument descriptors outside run.
// A weights handle must outlive its plans. Preparation is waited on once per
// plan/stream before its first run. Run only queues work;it does not synchronize
// or allocate device memory. Plans can share immutable weights. Host calls on a plan are
// serialized internally;destroy must not race with create/run on that handle.
// Plan destruction waits only streams on which that plan was launched.
// All calls use the same current HIP device as weights_create.
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_create_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 weights,
 const ck_tile_int8_nt_cached_group_desc_v1* groups,int32_t group_count,
 ck_tile_int8_nt_cached_plan_handle_v1* result);
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_query_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 plan,ck_tile_int8_nt_cached_plan_info_v1* info);
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_run_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 plan,void* stream);
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_destroy_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 plan);
CK_TILE_INT8_CACHED_API const char* ck_tile_int8_nt_cached_last_error_v1(void);
#ifdef __cplusplus
}
#endif
#endif
