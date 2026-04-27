import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="minisgl.shadowkv_kernels",
            sources=[
                "python/minisgl/kernel/shadowkv/csrc/torch_test.cpp",
            ],
            # include_dirs=[os.path.abspath("python/minisgl/kernels/shadowkv/csrs/include")],
            extra_compile_args={
            "cxx": ["-O3", "-DTORCH_USE_CUDA_DSA", "-w", '-lineinfo'],
            "nvcc": ["-O3",
                     "-DTORCH_USE_CUDA_DSA",
                     "-w",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extended-lambda",
                     "-Xptxas=-v"]
        },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

