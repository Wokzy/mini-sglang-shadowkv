"""Add packed active-slot entry points without changing upstream attention arithmetic."""


def function_body(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def add_active_api(root):
    lsh = root / "lsh/lsh.cc"
    header = root / "lsh/lsh.h"
    text = lsh.read_text()
    retrieve = function_body(text, "int LSH::retrieve(")
    retrieve = retrieve.replace("LSH::retrieve(", "LSH::retrieve_active(")
    retrieve = retrieve.replace("int head_id,", "int head_id, int row_id,")
    for old in ("query + head_id", "results + head_id", "mask + head_id"):
        retrieve = retrieve.replace(old, old.replace("head_id", "row_id"))
    batch = """void LSH::batch_retrieve_active(int layer_id, torch::Tensor slots_pt,
    torch::Tensor query_pt, torch::Tensor results_pt, torch::Tensor nnz_pt) {
    auto slots = slots_pt.data_ptr<int64_t>();
    const int count = slots_pt.numel();
    TORCH_CHECK(count <= this->batch_size, "Too many active slots");
    for (int i = 0; i < count; ++i)
        TORCH_CHECK(slots[i] >= 0 && slots[i] < this->batch_size, "Invalid active slot");
    auto query = query_pt.data_ptr<int>();
    auto results = results_pt.data_ptr<int>();
    auto nnz = nnz_pt.data_ptr<int>();
    #pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
    for (int row = 0; row < count * this->num_attention_heads; ++row) {
        const int head = slots[row / this->num_attention_heads] * this->num_attention_heads
                         + row % this->num_attention_heads;
        nnz[row] = this->retrieve_active(layer_id, head, row, query, results);
    }
}
"""
    text = text.replace("PYBIND11_MODULE(", retrieve + "\n" + batch + "\nPYBIND11_MODULE(")
    text = text.replace(
        '.def("batch_retrieve",',
        '.def("batch_retrieve_active", &LSH::batch_retrieve_active)\n        .def("batch_retrieve",',
    )
    lsh.write_text(
        text.replace(
            '.def("fill", &LSH::fill)',
            '.def("fill", &LSH::fill, py::call_guard<py::gil_scoped_release>())',
        )
    )
    declarations = retrieve.split("{")[0].replace("LSH::", "") + ";\n"
    declarations += batch.split("{")[0].replace("LSH::", "") + ";\n"
    header.write_text(header.read_text().replace("public:", "public:\n" + declarations))

    sparse = root / "sparse_attention/sparse_attention.cc"
    header = root / "sparse_attention/sparse_attention.h"
    text = sparse.read_text()
    functions, declarations = [], []
    for suffix in ("", "_bf16"):
        old = "dynamic_attention" + suffix
        new = "active_attention" + suffix
        func = function_body(text, "void SparseAttentionServer::" + old + "(")
        func = func.replace(old + "(", new + "(")
        func = func.replace("int layer_id,", "int layer_id, torch::Tensor slots_pt,")
        func = func.replace(
            "{",
            """{
    auto slots = slots_pt.data_ptr<int64_t>();
    const int count = slots_pt.numel();
    TORCH_CHECK(count <= this->batch_size, "Too many active slots");
    for (int j = 0; j < count; ++j)
        TORCH_CHECK(slots[j] >= 0 && slots[j] < this->batch_size, "Invalid active slot");
""",
            1,
        )
        func = func.replace(
            "i < this->batch_size * this->num_attention_heads",
            "i < count * this->num_attention_heads",
        )
        func = func.replace(
            "(i/this->num_attention_groups)",
            "((slots[i / this->num_attention_heads] * this->num_attention_heads + i % this->num_attention_heads) / this->num_attention_groups)",
        )
        declarations.append(func.split("{")[0].replace("SparseAttentionServer::", "") + ";\n")
        functions.append(
            ("#ifdef __AVX512BF16__\n" if suffix else "")
            + func
            + ("\n#endif\n" if suffix else "\n")
        )
    text = text.replace("PYBIND11_MODULE(", "\n".join(functions) + "\nPYBIND11_MODULE(")
    binding = """
#ifdef __AVX512BF16__
        .def("attention_active", &SparseAttentionServer::active_attention_bf16)
#else
        .def("attention_active", &SparseAttentionServer::active_attention)
#endif
"""
    text = text.replace('.def("attention_wrapper",', binding + '        .def("attention_wrapper",')
    text = text.replace(
        '.def("fill", &SparseAttentionServer::fill)',
        '.def("fill", &SparseAttentionServer::fill, py::call_guard<py::gil_scoped_release>())',
    )
    sparse.write_text(text)
    header.write_text(header.read_text().replace("public:", "public:\n" + "".join(declarations)))
