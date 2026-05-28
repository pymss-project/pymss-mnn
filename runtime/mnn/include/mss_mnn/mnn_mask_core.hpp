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

MNNBackend mnn_backend_from_name(const std::string& name);
std::string mnn_backend_name(MNNBackend backend);

struct MaskCoreOptions {
    std::string input_name = "stft_repr";
    std::string output_name = "mask";
    MNNBackend backend = MNNBackend::CPU;
    int threads = 1;
};

struct MNNTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
};

struct MNNModelOptions {
    MNNBackend backend = MNNBackend::CPU;
    int threads = 1;
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

    std::vector<float> run(const std::vector<float>& input, const std::vector<int>& input_shape);
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
