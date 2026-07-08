# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path

import ck4inductor
from ck4inductor._template_parser import (
    find_template_instances,
    parse_template_instances_from_file,
    split_template_args,
)
from ck4inductor.util import library_path
from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_library as gen_gemm_ops_library,
    gen_ops_preselected as gen_gemm_ops_preselected,
)
from ck4inductor.batched_universal_gemm.gen_instances import (
    gen_ops_library as gen_batched_gemm_ops_library,
)
from ck4inductor.grouped_conv_fwd.gen_instances import (
    gen_conv_ops_library as gen_grouped_conv_ops_library,
)
from ck4inductor.hcu_grouped_conv_fwd.gen_instances import (
    gen_conv_ops_library as gen_hcu_grouped_conv_ops_library,
)
from ck4inductor.ck_tile_universal_gemm.gen_instances import ops as gen_ck_tile_gemm_ops

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


def _load_pyproject():
    if tomllib is None:
        raise unittest.SkipTest("tomllib is unavailable")
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


class TestPackageSkeleton(unittest.TestCase):
    def test_ck4inductor_importable(self):
        self.assertTrue(ck4inductor.__version__())

    def test_library_path_points_to_readable_library(self):
        path = Path(library_path())
        self.assertTrue(path.is_dir())
        self.assertTrue((path / "src" / "tensor_operation_instance_hcu" / "gpu").is_dir())

    def test_package_data_covers_hcu_instance_paths(self):
        package_data = _load_pyproject()["tool"]["setuptools"]["package-data"]
        library_patterns = package_data["ck4inductor.library"]
        library_root = Path(__file__).resolve().parents[2] / "library"
        covered_paths = {
            path.relative_to(library_root).as_posix()
            for pattern in library_patterns
            for path in library_root.glob(pattern)
            if path.is_file()
        }
        required_paths = {
            "src/tensor_operation_instance_hcu/gpu/gemm_universal/"
            "device_gemm_xdl_universal_bf16_bf16_bf16.hpp",
            "src/tensor_operation_instance_hcu/gpu/gemm_universal_batched/"
            "device_batched_gemm_xdl_universal_bf16_bf16_bf16.hpp",
            "src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx928/"
            "nhwgc_gkyxc_nhwgk/"
            "device_grouped_conv2d_fwd_mmac_v2_nhwgc_gkyxc_nhwgk_f16_default_cshuffle_instance.cpp",
            "src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx928/"
            "nhwgc_gkyxc_nhwgk/device_grouped_conv2d_fwd_common.hpp",
            "src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx936/"
            "nhwgc_gkyxc_nhwgk/ck_tile/ck_tile_conv2d_v2r1_f16_f16_f1x1p0_instance.cpp",
        }
        self.assertLessEqual(required_paths, covered_paths)


class TestTemplateParser(unittest.TestCase):
    def setUp(self):
        self.library = Path(library_path())

    def test_split_template_args_keeps_nested_tokens(self):
        args = split_template_args(
            "S<1, 2>, Tuple<A, Sequence<3, 4>>, "
            "DeviceFoo<Bar<Baz, 5>, true>, true"
        )
        self.assertEqual(
            args,
            [
                "S<1, 2>",
                "Tuple<A, Sequence<3, 4>>",
                "DeviceFoo<Bar<Baz, 5>, true>",
                "true",
            ],
        )

    def test_parse_gemm_cpp_instance(self):
        path = self.library / "src" / "tensor_operation_instance_hcu" / "gpu" / "gemm_universal" / "device_gemm_xdl_universal_bf16_bf16_bf16.hpp"
        instances = parse_template_instances_from_file(path, "DeviceGemm_Xdl_CShuffleV3")
        self.assertTrue(instances)
        first = instances[0]
        self.assertEqual(len(first.args), 42)
        self.assertEqual(first.args[:3], ("Row", "Col", "Row"))
        self.assertEqual(first.args[22], "S<4, 64, 1>")
        self.assertEqual(first.args[-2:], ("BlockGemmPipelineScheduler::Intrawave", "BlockGemmPipelineVersion::v3"))

    def test_parse_batched_gemm_cpp_instance(self):
        path = self.library / "src" / "tensor_operation_instance_hcu" / "gpu" / "gemm_universal_batched" / "device_batched_gemm_xdl_universal_bf16_bf16_bf16.hpp"
        instances = parse_template_instances_from_file(path, "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3")
        self.assertTrue(instances)
        first = instances[0]
        self.assertEqual(len(first.args), 44)
        self.assertEqual(first.args[:4], ("Row", "Col", "Tuple<>", "Row"))
        self.assertEqual(first.args[4:8], ("BF16", "BF16", "Tuple<>", "BF16"))
        self.assertEqual(first.args[-2:], ("BlockGemmPipelineScheduler::Intrawave", "BlockGemmPipelineVersion::v3"))

    def test_parse_grouped_conv_hpp_instance(self):
        path = self.library / "src" / "tensor_operation_instance" / "gpu" / "grouped_conv2d_fwd" / "device_grouped_conv2d_fwd_xdl_instance.hpp"
        instances = parse_template_instances_from_file(path, "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle")
        self.assertTrue(instances)
        first = instances[0]
        self.assertEqual(len(first.args), 45)
        self.assertEqual(first.args[0], "2")
        self.assertEqual(first.args[1:5], ("ALayout", "BLayout", "DsLayout", "ELayout"))
        self.assertEqual(first.args[14:17], ("ConvSpec", "GemmMNKPadding", "1"))

    def test_find_template_instances_scans_with_pathlib(self):
        root = self.library / "src" / "tensor_operation_instance" / "gpu" / "grouped_conv2d_fwd"
        instances = find_template_instances(
            root,
            "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle",
            suffixes=(".hpp",),
        )
        self.assertTrue(instances)
        self.assertTrue(any(instance.path.name == "device_grouped_conv2d_fwd_xdl_instance.hpp" for instance in instances))

    def test_parse_hcu_grouped_conv_direct_mmac_instance(self):
        path = self.library / "src" / "tensor_operation_instance_hcu" / "gpu" / "grouped_conv2d_fwd_gfx928" / "nhwgc_gkyxc_nhwgk" / "device_grouped_conv2d_fwd_mmac_v2_nhwgc_gkyxc_nhwgk_f16_default_cshuffle_instance.cpp"
        instances = parse_template_instances_from_file(path, "DeviceGroupedConvFwd_mmac_v2_cshuffle")
        self.assertTrue(instances)
        first = instances[0]
        self.assertEqual(first.args[:4], ("2", "InLayout", "WeiLayout", "OutLayout"))
        self.assertIn("ConvFwdDefault", first.args)

    def test_parse_hcu_grouped_conv_ck_tile_instance(self):
        path = self.library / "src" / "tensor_operation_instance_hcu" / "gpu" / "grouped_conv2d_fwd_gfx936" / "nhwgc_gkyxc_nhwgk" / "ck_tile" / "ck_tile_conv2d_v2r1_f16_f16_f1x1p0_instance.cpp"
        instances = parse_template_instances_from_file(path, "Kernel")
        self.assertTrue(instances)
        first = instances[0]
        self.assertGreater(len(first.args), 5)



class TestUniversalGemmGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = gen_gemm_ops_library()

    def test_gen_gemm_instances(self):
        self.assertTrue(self.instances)
        first = self.instances[0]
        self.assertEqual(first.name(), first.name())
        lowered = first.name().lower()
        self.assertIn("shuffle_v3", lowered)
        self.assertIn("multid", lowered)
        self.assertTrue(first.name().startswith("ck_devicegemm_multid_xdl_shuffle_v3_"))
        self.assertEqual(first.ds_layouts, tuple())
        self.assertEqual(first.ds_element_dtypes, tuple())

    def test_gemm_instances_cover_f16_and_bf16(self):
        dtypes = {op.a_element_dtype for op in self.instances}
        self.assertIn("BF16", dtypes)
        self.assertTrue(any(op.block_gemm_pipeline_version == "BlockGemmPipelineVersion::v3" for op in self.instances))

    def test_preselected_gemm_is_disabled_until_dcu_benchmark(self):
        self.assertEqual(gen_gemm_ops_preselected(), [])


class TestBatchedUniversalGemmGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = gen_batched_gemm_ops_library()

    def test_gen_batched_gemm_instances(self):
        self.assertTrue(self.instances)
        first = self.instances[0]
        lowered = first.name().lower()
        self.assertIn("shuffle_v3", lowered)
        self.assertIn("multi_d", lowered)
        self.assertIn("c_shuffle_v3", lowered)
        self.assertTrue(first.name().startswith("ck_device_batched_gemm_multi_d_xdl_c_shuffle_v3_"))
        self.assertEqual(first.ds_layouts, tuple())
        self.assertEqual(first.ds_element_dtypes, tuple())
        self.assertEqual(first.c_shuffle_dtype, "BF16")

    def test_batched_gemm_instances_cover_hcu_bf16_rcr_layout(self):
        layouts = {(op.a_layout, op.b_layout, op.c_layout) for op in self.instances}
        self.assertIn(("Row", "Col", "Row"), layouts)
        self.assertTrue(any(op.a_element_dtype == "BF16" for op in self.instances))
        self.assertNotIn("AData", {op.a_element_dtype for op in self.instances})

class TestGroupedConvFwdGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = gen_grouped_conv_ops_library()

    def test_gen_grouped_conv_instances(self):
        self.assertTrue(self.instances)
        first = self.instances[0]
        lowered = first.name().lower()
        self.assertNotIn("multipleabd", lowered)
        self.assertNotIn("shuffle_v3", lowered)
        self.assertTrue(first.name().startswith("ck_device_grouped_conv_fwd_multiple_d_xdl_cshuffle_"))
        self.assertEqual(first.ds_layout, tuple())
        self.assertEqual(first.ds_element_dtype, tuple())
        self.assertEqual(first.loop_scheduler, "LoopScheduler::Default")

    def test_grouped_conv_instances_cover_standard_layouts(self):
        layouts = {(op.a_layout, op.b_layout, op.e_layout) for op in self.instances}
        self.assertIn(("NHWGC", "GKYXC", "NHWGK"), layouts)
        self.assertIn(("GNHWC", "GKYXC", "GNHWK"), layouts)
        dtypes = {op.a_element_dtype for op in self.instances}
        self.assertIn("F16", dtypes)
        self.assertIn("BF16", dtypes)
        self.assertIn("F32", dtypes)
        self.assertNotIn("DsDatatype", {str(op.ds_element_dtype) for op in self.instances})


class TestHCUGroupedConvFwdGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = gen_hcu_grouped_conv_ops_library()

    def test_gen_hcu_grouped_conv_instances(self):
        self.assertTrue(self.instances)
        first = self.instances[0]
        self.assertTrue(first.name().startswith("ck_hcu_grouped_conv_fwd_"))
        self.assertTrue(first.source_path.startswith("grouped_conv2d_fwd_gfx"))
        self.assertNotEqual(first.template_name, "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle")
        self.assertTrue(first.raw_args)

    def test_hcu_grouped_conv_instances_cover_hcu_arches_and_families(self):
        arches = {op.arch for op in self.instances}
        self.assertIn("gfx928", arches)
        self.assertIn("gfx936", arches)
        families = {op.family for op in self.instances}
        self.assertIn("mmac_v2_cshuffle", families)
        self.assertTrue(any(family.startswith("ck_tile") for family in families))

    def test_hcu_grouped_conv_instances_cover_layouts_and_specs(self):
        layouts = {(op.a_layout, op.b_layout, op.e_layout) for op in self.instances}
        self.assertIn(("NHWGC", "GKYXC", "NHWGK"), layouts)
        self.assertIn(("NGCHW", "GKCYX", "NGKHW"), layouts)
        specs = {op.conv_forward_specialization for op in self.instances}
        self.assertIn("ConvFwdDefault", specs)
        self.assertIn("ConvFwd1x1P0", specs)
        dtypes = {op.a_element_dtype for op in self.instances}
        self.assertIn("F16", dtypes)


class TestCKTileUniversalGemmGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = gen_ck_tile_gemm_ops()

    def test_ck_tile_ops_are_allowlisted(self):
        self.assertTrue(self.instances)
        pipelines = {op.pipeline for op in self.instances}
        self.assertEqual(pipelines, {"Mem", "CompV4"})
        self.assertNotIn("CompV3", pipelines)
        self.assertEqual({op.scheduler for op in self.instances}, {"Intrawave"})
        self.assertLessEqual({op.epilogue for op in self.instances}, {"Default", "CShuffle"})

    def test_ck_tile_names_are_stable(self):
        first = self.instances[0]
        self.assertEqual(first.name(), first.name())
        self.assertTrue(first.name().startswith("ck_tile_gemm_universal_"))
        self.assertIn(first.pipeline, first.name())

if __name__ == "__main__":
    unittest.main()
