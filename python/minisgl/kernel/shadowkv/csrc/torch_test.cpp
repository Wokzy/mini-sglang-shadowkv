#include <torch/extension.h>

void fill_ones(at::Tensor x) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float);
    float* ptr = (float*)x.data_ptr();

    for (int i = 0; i < x.size(0); i++) {
        for (int j = 0; j < x.size(1); j++) {
            ptr[i * x.stride(0) + j * x.stride(1)] = 1.0;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_ones", &fill_ones, "fill_ones");
}
