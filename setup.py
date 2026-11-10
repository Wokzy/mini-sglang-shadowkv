import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

COMPUTE_CAP = os.environ.get("COMPUTE_CAP", "90")
if int(COMPUTE_CAP) >= 90:
    COMPUTE_CAP += "a"


ext_modules = [
    CUDAExtension(
        name="minisgl.shadowkv_kernels",
        sources=[
            "python/minisgl/kernel/shadowkv/csrc/lib.cpp",
            "python/minisgl/kernel/higgs/csrc/higgs_quantizer.cpp",
            "python/minisgl/kernel/higgs/csrc/higgs_quantizer_cuda.cu",
            "python/minisgl/kernel/higgs/csrc/higgs_quantization_cuda.cu",
            "python/minisgl/kernel/higgs/csrc/higgs_dequantization_full.cu",
            "python/minisgl/kernel/higgs/csrc/higgs_score.cu",
            "python/minisgl/kernel/higgs/csrc/higgs_quantize_heads.cu",
            "python/minisgl/kernel/higgs/csrc/higgs_dequantize_heads.cu",
            "python/minisgl/kernel/shadowkv/csrc/gather_kv_cache.cu",
            "python/minisgl/kernel/shadowkv/csrc/metadata_kernels.cu",
            "python/minisgl/kernel/shadowkv/csrc/shadowkv.cpp",
        ],
        # include_dirs=[os.path.abspath("python/minisgl/kernels/shadowkv/csrc/include")],
        extra_compile_args={
            "cxx": ["-O3", "-DTORCH_USE_CUDA_DSA", "-w", "-lineinfo"],
            "nvcc": [
                "-O3",
                "-DTORCH_USE_CUDA_DSA",
                "-w",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                f'-gencode=arch=compute_{COMPUTE_CAP},code="sm_{COMPUTE_CAP}"',
                "-Xptxas=-v",
            ],
        },
    )
]


if not os.environ.get("DISABLE_SHADOWKV", False):
    """
    Reference: https://github.com/ByteDance-Seed/ShadowKV/blob/main/setup.py
    """
    ext_modules.append(
        CUDAExtension(
            name="minisgl.og_shadowkv_kernels",
            sources=[
                "python/minisgl/og_shadowkv/kernels/main.cu",
                "python/minisgl/og_shadowkv/kernels/rope.cu",
                "python/minisgl/og_shadowkv/kernels/rope_new.cu",
                "python/minisgl/og_shadowkv/kernels/gather_copy.cu",
                "python/minisgl/og_shadowkv/kernels/batch_gather_gemm.cu",
                "python/minisgl/og_shadowkv/kernels/batch_gemm_softmax.cu",
            ],
            include_dirs=[
                os.path.join(ROOT_DIR, "3rdparty/cutlass/include"),
                os.path.join(ROOT_DIR, "3rdparty/cutlass/examples/common"),
                os.path.join(ROOT_DIR, "3rdparty/cutlass/tools/util/include"),
                os.path.join(ROOT_DIR, "python/minisgl/og_shadowkv/kernels"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lcuda",
                    "-lcudart",
                    f'-gencode=arch=compute_{COMPUTE_CAP},code="sm_{COMPUTE_CAP}"',
                    "-Xptxas=-v",
                ],
            },
        )
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
