#include <ATen/mps/MPSAllocatorInterface.h>
#include <torch/extension.h>

#include <cstdint>

pybind11::dict mps_shared_buffer_info(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::MPS, "expected an MPS tensor");
    TORCH_CHECK(tensor.is_contiguous(), "expected a contiguous tensor");

    auto* allocator = at::mps::getIMPSAllocator();
    const void* data = tensor.data_ptr();
    auto shared = allocator->getSharedBufferPtr(data);

    pybind11::dict out;
    out["data_ptr"] = pybind11::int_(reinterpret_cast<std::uintptr_t>(data));
    out["shared_buffer_ptr"] = pybind11::int_(reinterpret_cast<std::uintptr_t>(shared.first));
    out["shared_buffer_offset"] = pybind11::int_(shared.second);
    out["is_shared_buffer"] = pybind11::bool_(allocator->isSharedBuffer(data));
    out["shared_storage_supported"] = pybind11::bool_(allocator->isSharedStorageSupported());
    out["buffer_id"] = pybind11::int_(allocator->getBufferId(data));
    out["unaligned_buffer_size"] = pybind11::int_(allocator->getUnalignedBufferSize(data));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mps_shared_buffer_info", &mps_shared_buffer_info);
}
