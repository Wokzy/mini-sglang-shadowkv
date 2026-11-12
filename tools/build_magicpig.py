#!/usr/bin/env python3
"""Build pinned upstream CPU kernels against the current PyTorch ABI.

Only a private source copy is patched. No PyTorch/CUDA dependencies are installed.
"""

import argparse
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from magicpig_active import add_active_api

REVISION = "ac9aa36c866330ca6ad2ce342a7848d7df6f49bb"


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream source in {path}: {old!r}")
    path.write_text(text.replace(old, new))


def prepare_sources(destination, threads, source=None):
    if destination.exists():
        raise ValueError(f"Build source directory already exists: {destination}")
    if source is not None:
        shutil.copytree(source / "library", destination)
    else:
        url = f"https://codeload.github.com/Infini-AI-Lab/MagicPIG/tar.gz/{REVISION}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            prefix = f"MagicPIG-{REVISION}/library/"
            for member in tar.getmembers():
                if not member.name.startswith(prefix) or not member.isfile():
                    continue
                relative = Path(member.name[len(prefix) :])
                if ".." in relative.parts or relative.is_absolute():
                    raise ValueError("Unsafe archive path")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    lsh = destination / "lsh/lsh.cc"
    sparse = destination / "sparse_attention/sparse_attention.cc"
    # Upstream fill appends bucket counts; serving must support reusing one request slot.
    replace_once(
        lsh,
        "int16_t * sorted_hash_code =",
        """memset(this->table_start[layer_id] + request_id * stride, 0, stride * sizeof(int));
memset(this->table_end[layer_id] + request_id * stride, 0, stride * sizeof(int));
int16_t * sorted_hash_code =""",
    )
    # An empty LSH sample is valid: CPU contributes zero mass to the GPU/CPU merge.
    replace_once(
        sparse,
        "void softmax_kernel(\nfloat *score, \nconst int nnz,\nfloat *max_value,\nfloat *expsum) {",
        """void softmax_kernel(
float *score,
const int nnz,
float *max_value,
float *expsum) {
    if (nnz == 0) {
        max_value[0] = -INFINITY;
        expsum[0] = -INFINITY;
        return;
    }""",
    )
    replace_once(
        sparse,
        "float theta = acosf(score[j] / norm);",
        """float cosine = norm > 0 ? score[j] / norm : 0.0f;
        float theta = acosf(fmaxf(-1.0f, fminf(1.0f, cosine)));""",
    )
    for filename, original, module in (
        (lsh, "lsh", "minisgl_magicpig_lsh"),
        (sparse, "sparse_attention_cpu", "minisgl_magicpig_sparse"),
    ):
        replace_once(
            filename,
            f"PYBIND11_MODULE({original}, m) {{",
            f'PYBIND11_MODULE({module}, m) {{\n    m.attr("adapter_version") = 2;',
        )
    for path in (destination / "lsh/lsh.h", destination / "sparse_attention/sparse_attention.h"):
        path.write_text(path.read_text().replace("THREADS 64", f"THREADS {threads}"))
    sparse.write_text(sparse.read_text().replace("num_threads(64)", f"num_threads({threads})"))

    add_active_api(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument(
        "--source", type=Path, help="Offline checkout of the pinned upstream revision"
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable AVX512_BF16 (default: detect build-host CPU; override for a different target)",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.bf16 is None:
        cpuinfo = Path("/proc/cpuinfo")
        args.bf16 = cpuinfo.exists() and "avx512_bf16" in cpuinfo.read_text().split()
    print(f"MagicPIG build: threads={args.threads}, AVX512_BF16={args.bf16}", flush=True)
    root = args.build_dir.resolve()
    prepare_sources(root / "sources", args.threads, args.source)
    # BuildExtension selects the installed torch ABI; upstream hardcodes the old ABI.
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    source_root = root / "sources"
    sparse_root = source_root / "sparse_attention"
    flags = ["-O3", "-mavx512f", "-fopenmp"] + (["-mavx512bf16"] if args.bf16 else [])
    thirdparty = sparse_root / "3rdparty/FBGEMM"
    sparse_sources = [str(sparse_root / "sparse_attention.cc")] + [
        str(thirdparty / "src" / f"{name}.cc")
        for name in (
            "FbgemmBfloat16Convert",
            "FbgemmBfloat16ConvertAvx2",
            "FbgemmBfloat16ConvertAvx512",
            "RefImplementations",
            "Utils",
        )
    ]
    os.environ.setdefault("MAX_JOBS", "2")
    setup(
        name="minisgl-magicpig-kernels",
        version="0.1.0",
        ext_modules=[
            CppExtension(
                "minisgl_magicpig_lsh",
                [str(source_root / "lsh/lsh.cc")],
                extra_compile_args=flags,
                extra_link_args=["-fopenmp"],
            ),
            CppExtension(
                "minisgl_magicpig_sparse",
                sparse_sources,
                include_dirs=[str(thirdparty / "include")],
                extra_compile_args=flags,
                extra_link_args=["-fopenmp"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        script_args=[
            "build_ext",
            "--build-lib",
            str(root / "lib"),
            "--build-temp",
            str(root / "temp"),
        ],
    )
    print(f"Add {root / 'lib'} to PYTHONPATH when launching MiniSGL.")


if __name__ == "__main__":
    main()
