#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mss_mnn {

enum class MNNBackend {
    CPU,
    Auto,
    Metal,
    OpenCL,
    Vulkan,
};

enum class MNNPrecision {
    Auto,
    Normal,
    High,
    Low,
    LowBF16,
};

MNNBackend mnn_backend_from_name(const std::string& name);
std::string mnn_backend_name(MNNBackend backend);
MNNPrecision mnn_precision_from_name(const std::string& name);
std::string mnn_precision_name(MNNPrecision precision);

struct MaskCoreOptions {
    std::string input_name = "stft_repr";
    std::string output_name = "mask";
    MNNBackend backend = MNNBackend::CPU;
    MNNPrecision precision = MNNPrecision::Auto;
    int threads = 1;
    int attention_option = 0;
    bool profile_ops = false;
    int profile_op_runs = 1;
    int profile_op_top_n = 20;
};

struct MNNTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
};

struct MNNModelOptions {
    MNNBackend backend = MNNBackend::CPU;
    MNNPrecision precision = MNNPrecision::Auto;
    int threads = 1;
    int attention_option = 0;
};

struct MNNRunProfile {
    double resize_ms = 0.0;
    double input_copy_ms = 0.0;
    double run_ms = 0.0;
    double output_copy_ms = 0.0;
    struct OpProfile {
        std::string name;
        std::string type;
        std::string input_shapes;
        std::string output_shapes;
        double total_ms = 0.0;
        int calls = 0;
        float flops = 0.0f;
    };
    std::vector<OpProfile> ops;
    std::vector<OpProfile> op_names;
};

class MNNModel {
public:
    MNNModel(const std::string& model_path, MNNModelOptions options = {});
    ~MNNModel();

    MNNModel(const MNNModel&) = delete;
    MNNModel& operator=(const MNNModel&) = delete;
    MNNModel(MNNModel&&) noexcept;
    MNNModel& operator=(MNNModel&&) noexcept;

    std::vector<MNNTensor> run(const std::vector<MNNTensor>& inputs, const std::vector<std::string>& output_names);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MNNMaskCore {
public:
    MNNMaskCore(const std::string& model_path, MaskCoreOptions options = {});
    ~MNNMaskCore();

    MNNMaskCore(const MNNMaskCore&) = delete;
    MNNMaskCore& operator=(const MNNMaskCore&) = delete;
    MNNMaskCore(MNNMaskCore&&) noexcept;
    MNNMaskCore& operator=(MNNMaskCore&&) noexcept;

    std::vector<float> run(const std::vector<float>& input, const std::vector<int>& input_shape, MNNRunProfile* profile = nullptr);
    const std::vector<int>& output_shape() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> read_f32_file(const std::string& path);
void write_f32_file(const std::string& path, const std::vector<float>& data);
std::vector<int> parse_shape(const std::string& value);
std::size_t shape_element_count(const std::vector<int>& shape);

}  // namespace mss_mnn
