#include "mss_mnn/mnn_mask_core.hpp"

#include "apple_autorelease_pool.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <fstream>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mss_mnn {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

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

MNN::BackendConfig::PrecisionMode precision_type(MNNBackend backend, MNNPrecision precision) {
    if (precision == MNNPrecision::Auto) {
        const bool metal_capable_backend = backend == MNNBackend::Metal || backend == MNNBackend::Auto;
        precision = metal_capable_backend ? MNNPrecision::High : MNNPrecision::Normal;
    }
    switch (precision) {
        case MNNPrecision::Auto:
            return MNN::BackendConfig::Precision_Normal;
        case MNNPrecision::Normal:
            return MNN::BackendConfig::Precision_Normal;
        case MNNPrecision::High:
            return MNN::BackendConfig::Precision_High;
        case MNNPrecision::Low:
            return MNN::BackendConfig::Precision_Low;
        case MNNPrecision::LowBF16:
            return MNN::BackendConfig::Precision_Low_BF16;
    }
    return MNN::BackendConfig::Precision_Normal;
}

void apply_session_hints(MNN::Interpreter& interpreter, int attention_option) {
    if (attention_option > 0) {
        interpreter.setSessionHint(MNN::Interpreter::ATTENTION_OPTION, attention_option);
    }
}

std::string tensor_shapes(const std::vector<MNN::Tensor*>& tensors) {
    std::ostringstream stream;
    stream << "[";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << "[";
        if (tensors[i]) {
            const auto shape = tensors[i]->shape();
            for (std::size_t d = 0; d < shape.size(); ++d) {
                if (d > 0) {
                    stream << "x";
                }
                stream << shape[d];
            }
        }
        stream << "]";
    }
    stream << "]";
    return stream.str();
}

}  // namespace

struct MNNMaskCore::Impl {
    MaskCoreOptions options;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int profiled_op_runs = 0;
};

struct InputMetadata {
    std::string name;
    std::vector<int> shape;
};

struct MNNModel::Impl {
    MNNModelOptions options;
    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session* session = nullptr;
    std::vector<InputMetadata> input_metadata;
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

MNNPrecision mnn_precision_from_name(const std::string& name) {
    std::string value = name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "normal") {
        return MNNPrecision::Normal;
    }
    if (value == "auto") {
        return MNNPrecision::Auto;
    }
    if (value == "high") {
        return MNNPrecision::High;
    }
    if (value == "low") {
        return MNNPrecision::Low;
    }
    if (value == "low-bf16" || value == "low_bf16" || value == "bf16") {
        return MNNPrecision::LowBF16;
    }
    throw std::runtime_error("unknown MNN precision: " + name);
}

std::string mnn_precision_name(MNNPrecision precision) {
    switch (precision) {
        case MNNPrecision::Auto:
            return "auto";
        case MNNPrecision::Normal:
            return "normal";
        case MNNPrecision::High:
            return "high";
        case MNNPrecision::Low:
            return "low";
        case MNNPrecision::LowBF16:
            return "low-bf16";
    }
    return "normal";
}

MNNModel::MNNModel(const std::string& model_path, MNNModelOptions options)
    : impl_(std::make_unique<Impl>()) {
    ScopedAutoreleasePool autorelease_pool;
    impl_->options = std::move(options);
    impl_->interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!impl_->interpreter) {
        throw std::runtime_error("failed to create MNN interpreter for " + model_path);
    }
    apply_session_hints(*impl_->interpreter, impl_->options.attention_option);

    MNN::ScheduleConfig config;
    MNN::BackendConfig backend_config;
    backend_config.precision = precision_type(impl_->options.backend, impl_->options.precision);
    config.type = backend_type(impl_->options.backend);
    config.numThread = impl_->options.threads;
    config.backendConfig = &backend_config;
    impl_->session = impl_->interpreter->createSession(config);
    if (!impl_->session) {
        throw std::runtime_error("failed to create MNN session");
    }
}

MNNModel::~MNNModel() = default;
MNNModel::MNNModel(MNNModel&&) noexcept = default;
MNNModel& MNNModel::operator=(MNNModel&&) noexcept = default;

std::vector<MNNTensor> MNNModel::run(const std::vector<MNNTensor>& inputs, const std::vector<std::string>& output_names) {
    ScopedAutoreleasePool autorelease_pool;
    if (inputs.empty()) {
        throw std::runtime_error("MNNModel::run requires at least one input");
    }
    if (output_names.empty()) {
        throw std::runtime_error("MNNModel::run requires at least one output name");
    }

    bool needs_resize = impl_->input_metadata.size() != inputs.size();
    if (needs_resize) {
        impl_->input_metadata.resize(inputs.size());
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (input.data.size() != shape_element_count(input.shape)) {
            throw std::runtime_error("input size does not match input shape: " + input.name);
        }
        auto* input_tensor = impl_->interpreter->getSessionInput(impl_->session, input.name.c_str());
        if (!input_tensor) {
            throw std::runtime_error("missing MNN input tensor: " + input.name);
        }
        if (needs_resize || impl_->input_metadata[i].name != input.name || impl_->input_metadata[i].shape != input.shape) {
            impl_->interpreter->resizeTensor(input_tensor, input.shape);
            impl_->input_metadata[i].name = input.name;
            impl_->input_metadata[i].shape = input.shape;
            needs_resize = true;
        }
    }
    if (needs_resize) {
        impl_->interpreter->resizeSession(impl_->session);
    }

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
    ScopedAutoreleasePool autorelease_pool;
    impl_->options = std::move(options);
    impl_->interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!impl_->interpreter) {
        throw std::runtime_error("failed to create MNN interpreter for " + model_path);
    }
    apply_session_hints(*impl_->interpreter, impl_->options.attention_option);

    MNN::ScheduleConfig config;
    MNN::BackendConfig backend_config;
    backend_config.precision = precision_type(impl_->options.backend, impl_->options.precision);
    config.type = backend_type(impl_->options.backend);
    config.numThread = impl_->options.threads;
    config.backendConfig = &backend_config;
    impl_->session = impl_->interpreter->createSession(config);
    if (!impl_->session) {
        throw std::runtime_error("failed to create MNN session");
    }
}

MNNMaskCore::~MNNMaskCore() = default;
MNNMaskCore::MNNMaskCore(MNNMaskCore&&) noexcept = default;
MNNMaskCore& MNNMaskCore::operator=(MNNMaskCore&&) noexcept = default;

std::vector<float> MNNMaskCore::run(const std::vector<float>& input, const std::vector<int>& input_shape, MNNRunProfile* profile) {
    ScopedAutoreleasePool autorelease_pool;
    if (input.size() != shape_element_count(input_shape)) {
        throw std::runtime_error("input size does not match input shape");
    }
    MNNRunProfile local_profile;

    auto* input_tensor = impl_->interpreter->getSessionInput(impl_->session, impl_->options.input_name.c_str());
    if (!input_tensor) {
        throw std::runtime_error("missing MNN input tensor: " + impl_->options.input_name);
    }
    if (impl_->input_shape != input_shape) {
        const auto start = std::chrono::steady_clock::now();
        impl_->interpreter->resizeTensor(input_tensor, input_shape);
        impl_->interpreter->resizeSession(impl_->session);
        impl_->input_shape = input_shape;
        local_profile.resize_ms += elapsed_ms(start);
    }

    {
        const auto start = std::chrono::steady_clock::now();
        MNN::Tensor host_input(input_tensor, input_tensor->getDimensionType());
        std::copy(input.begin(), input.end(), host_input.host<float>());
        input_tensor->copyFromHostTensor(&host_input);
        local_profile.input_copy_ms += elapsed_ms(start);
    }

    const auto run_start = std::chrono::steady_clock::now();
    MNN::ErrorCode code = MNN::NO_ERROR;
    const bool profile_ops = impl_->options.profile_ops && impl_->profiled_op_runs < impl_->options.profile_op_runs;
    if (profile_ops) {
        struct MutableOpProfile {
            std::string name;
            std::string type;
            std::string input_shapes;
            std::string output_shapes;
            double total_ms = 0.0;
            int calls = 0;
            float flops = 0.0f;
        };
        std::unordered_map<std::string, MutableOpProfile> op_type_profiles;
        std::unordered_map<std::string, MutableOpProfile> op_name_profiles;
        auto op_start = std::chrono::steady_clock::now();
        std::string current_input_shapes;
        auto before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo*) {
            op_start = std::chrono::steady_clock::now();
            if (impl_->options.profile_op_top_n > 0) {
                current_input_shapes = tensor_shapes(tensors);
            }
            return true;
        };
        auto after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            const std::string type = info ? info->type() : "unknown";
            const std::string name = info ? info->name() : "";
            const double ms = elapsed_ms(op_start);
            auto& type_item = op_type_profiles[type];
            type_item.type = type;
            type_item.total_ms += ms;
            type_item.calls += 1;
            if (info) {
                type_item.flops += info->flops();
            }
            if (impl_->options.profile_op_top_n > 0) {
                auto& name_item = op_name_profiles[type + "\n" + name];
                name_item.type = type;
                name_item.name = name.empty() ? "<unnamed>" : name;
                if (name_item.input_shapes.empty()) {
                    name_item.input_shapes = current_input_shapes;
                    name_item.output_shapes = tensor_shapes(tensors);
                }
                name_item.total_ms += ms;
                name_item.calls += 1;
                if (info) {
                    name_item.flops += info->flops();
                }
            }
            return true;
        };
        code = impl_->interpreter->runSessionWithCallBackInfo(impl_->session, before, after, true);
        ++impl_->profiled_op_runs;
        local_profile.ops.reserve(op_type_profiles.size());
        for (const auto& item : op_type_profiles) {
            MNNRunProfile::OpProfile op;
            op.type = item.second.type;
            op.total_ms = item.second.total_ms;
            op.calls = item.second.calls;
            op.flops = item.second.flops;
            local_profile.ops.push_back(std::move(op));
        }
        local_profile.op_names.reserve(op_name_profiles.size());
        for (const auto& item : op_name_profiles) {
            MNNRunProfile::OpProfile op;
            op.name = item.second.name;
            op.type = item.second.type;
            op.input_shapes = item.second.input_shapes;
            op.output_shapes = item.second.output_shapes;
            op.total_ms = item.second.total_ms;
            op.calls = item.second.calls;
            op.flops = item.second.flops;
            local_profile.op_names.push_back(std::move(op));
        }
        std::sort(local_profile.op_names.begin(), local_profile.op_names.end(), [](const auto& a, const auto& b) {
            return a.total_ms > b.total_ms;
        });
        const auto top_n = static_cast<std::size_t>(std::max(0, impl_->options.profile_op_top_n));
        if (local_profile.op_names.size() > top_n) {
            local_profile.op_names.resize(top_n);
        }
    } else {
        code = impl_->interpreter->runSession(impl_->session);
    }
    local_profile.run_ms += elapsed_ms(run_start);
    if (code != 0) {
        throw std::runtime_error("MNN runSession failed");
    }

    const auto output_start = std::chrono::steady_clock::now();
    auto* output_tensor = impl_->interpreter->getSessionOutput(impl_->session, impl_->options.output_name.c_str());
    if (!output_tensor) {
        throw std::runtime_error("missing MNN output tensor: " + impl_->options.output_name);
    }
    impl_->output_shape = output_tensor->shape();
    MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
    output_tensor->copyToHostTensor(&host_output);

    const auto count = shape_element_count(impl_->output_shape);
    const float* ptr = host_output.host<float>();
    std::vector<float> output(ptr, ptr + count);
    local_profile.output_copy_ms += elapsed_ms(output_start);
    if (profile) {
        *profile = local_profile;
    }
    return output;
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
