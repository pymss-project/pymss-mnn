#include "mss_mnn/mnn_mask_core.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <fstream>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace mss_mnn {
namespace {

MNNForwardType backend_type(MNNBackend backend) {
    switch (backend) {
        case MNNBackend::CPU:
            return MNN_FORWARD_CPU;
        case MNNBackend::Auto:
            return MNN_FORWARD_AUTO;
        case MNNBackend::Metal:
            return MNN_FORWARD_METAL;
        case MNNBackend::OpenCL:
            return MNN_FORWARD_OPENCL;
        case MNNBackend::Vulkan:
            return MNN_FORWARD_VULKAN;
    }
    return MNN_FORWARD_CPU;
}

}  // namespace

struct MNNMaskCore::Impl {
    MaskCoreOptions options;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    std::vector<int> output_shape;
};

struct MNNModel::Impl {
    MNNModelOptions options;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
};

MNNBackend mnn_backend_from_name(const std::string& name) {
    std::string value = name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "cpu") {
        return MNNBackend::CPU;
    }
    if (value == "auto") {
        return MNNBackend::Auto;
    }
    if (value == "metal") {
        return MNNBackend::Metal;
    }
    if (value == "opencl") {
        return MNNBackend::OpenCL;
    }
    if (value == "vulkan") {
        return MNNBackend::Vulkan;
    }
    throw std::runtime_error("unknown MNN backend: " + name);
}

std::string mnn_backend_name(MNNBackend backend) {
    switch (backend) {
        case MNNBackend::CPU:
            return "cpu";
        case MNNBackend::Auto:
            return "auto";
        case MNNBackend::Metal:
            return "metal";
        case MNNBackend::OpenCL:
            return "opencl";
        case MNNBackend::Vulkan:
            return "vulkan";
    }
    return "cpu";
}

MNNModel::MNNModel(const std::string& model_path, MNNModelOptions options)
    : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
    impl_->interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!impl_->interpreter) {
        throw std::runtime_error("failed to create MNN interpreter for " + model_path);
    }

    MNN::ScheduleConfig config;
    config.type = backend_type(impl_->options.backend);
    config.numThread = impl_->options.threads;
    impl_->session = impl_->interpreter->createSession(config);
    if (!impl_->session) {
        throw std::runtime_error("failed to create MNN session");
    }
}

MNNModel::~MNNModel() = default;
MNNModel::MNNModel(MNNModel&&) noexcept = default;
MNNModel& MNNModel::operator=(MNNModel&&) noexcept = default;

std::vector<MNNTensor> MNNModel::run(const std::vector<MNNTensor>& inputs, const std::vector<std::string>& output_names) {
    if (inputs.empty()) {
        throw std::runtime_error("MNNModel::run requires at least one input");
    }
    if (output_names.empty()) {
        throw std::runtime_error("MNNModel::run requires at least one output name");
    }

    for (const auto& input : inputs) {
        if (input.data.size() != shape_element_count(input.shape)) {
            throw std::runtime_error("input size does not match input shape: " + input.name);
        }
        auto* input_tensor = impl_->interpreter->getSessionInput(impl_->session, input.name.c_str());
        if (!input_tensor) {
            throw std::runtime_error("missing MNN input tensor: " + input.name);
        }
        impl_->interpreter->resizeTensor(input_tensor, input.shape);
    }
    impl_->interpreter->resizeSession(impl_->session);

    for (const auto& input : inputs) {
        auto* input_tensor = impl_->interpreter->getSessionInput(impl_->session, input.name.c_str());
        MNN::Tensor host_input(input_tensor, input_tensor->getDimensionType());
        std::copy(input.data.begin(), input.data.end(), host_input.host<float>());
        input_tensor->copyFromHostTensor(&host_input);
    }

    auto code = impl_->interpreter->runSession(impl_->session);
    if (code != 0) {
        throw std::runtime_error("MNN runSession failed");
    }

    std::vector<MNNTensor> outputs;
    outputs.reserve(output_names.size());
    for (const auto& name : output_names) {
        auto* output_tensor = impl_->interpreter->getSessionOutput(impl_->session, name.c_str());
        if (!output_tensor) {
            throw std::runtime_error("missing MNN output tensor: " + name);
        }
        MNNTensor output;
        output.name = name;
        output.shape = output_tensor->shape();
        MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
        output_tensor->copyToHostTensor(&host_output);
        const auto count = shape_element_count(output.shape);
        const float* ptr = host_output.host<float>();
        output.data.assign(ptr, ptr + count);
        outputs.push_back(std::move(output));
    }
    return outputs;
}

MNNMaskCore::MNNMaskCore(const std::string& model_path, MaskCoreOptions options)
    : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
    impl_->interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!impl_->interpreter) {
        throw std::runtime_error("failed to create MNN interpreter for " + model_path);
    }

    MNN::ScheduleConfig config;
    config.type = backend_type(impl_->options.backend);
    config.numThread = impl_->options.threads;
    impl_->session = impl_->interpreter->createSession(config);
    if (!impl_->session) {
        throw std::runtime_error("failed to create MNN session");
    }
}

MNNMaskCore::~MNNMaskCore() = default;
MNNMaskCore::MNNMaskCore(MNNMaskCore&&) noexcept = default;
MNNMaskCore& MNNMaskCore::operator=(MNNMaskCore&&) noexcept = default;

std::vector<float> MNNMaskCore::run(const std::vector<float>& input, const std::vector<int>& input_shape) {
    if (input.size() != shape_element_count(input_shape)) {
        throw std::runtime_error("input size does not match input shape");
    }

    auto* input_tensor = impl_->interpreter->getSessionInput(impl_->session, impl_->options.input_name.c_str());
    if (!input_tensor) {
        throw std::runtime_error("missing MNN input tensor: " + impl_->options.input_name);
    }
    impl_->interpreter->resizeTensor(input_tensor, input_shape);
    impl_->interpreter->resizeSession(impl_->session);

    MNN::Tensor host_input(input_tensor, input_tensor->getDimensionType());
    std::copy(input.begin(), input.end(), host_input.host<float>());
    input_tensor->copyFromHostTensor(&host_input);

    auto code = impl_->interpreter->runSession(impl_->session);
    if (code != 0) {
        throw std::runtime_error("MNN runSession failed");
    }

    auto* output_tensor = impl_->interpreter->getSessionOutput(impl_->session, impl_->options.output_name.c_str());
    if (!output_tensor) {
        throw std::runtime_error("missing MNN output tensor: " + impl_->options.output_name);
    }
    impl_->output_shape = output_tensor->shape();
    MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
    output_tensor->copyToHostTensor(&host_output);

    const auto count = shape_element_count(impl_->output_shape);
    const float* ptr = host_output.host<float>();
    return std::vector<float>(ptr, ptr + count);
}

const std::vector<int>& MNNMaskCore::output_shape() const {
    return impl_->output_shape;
}

std::vector<float> read_f32_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff bytes = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("input file is not a float32 blob: " + path);
    }
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

void write_f32_file(const std::string& path, const std::vector<float>& data) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

std::vector<int> parse_shape(const std::string& value) {
    std::vector<int> shape;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (!part.empty()) {
            shape.push_back(std::stoi(part));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("shape must not be empty");
    }
    return shape;
}

std::size_t shape_element_count(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, [](std::size_t acc, int dim) {
        if (dim <= 0) {
            throw std::runtime_error("shape dimensions must be positive");
        }
        return acc * static_cast<std::size_t>(dim);
    });
}

}  // namespace mss_mnn
