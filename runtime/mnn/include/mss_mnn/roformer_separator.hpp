#pragma once

#include "mss_mnn/audio.hpp"
#include "mss_mnn/mnn_mask_core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mss_mnn {

struct RoformerMetadata {
    std::string preset;
    std::string model_type;
    int sample_rate = 44100;
    int chunk_size = 0;
    int overlap_size = 0;
    int n_fft = 2048;
    int hop_length = 512;
    int win_length = 2048;
    int stems = 1;
    int input_freq_channels = 0;
    int output_freq_channels = 0;
    int full_freq_channels = 0;
    int frames = 0;
    std::string mask_mode = "no_segm";
    std::vector<std::string> source_names;
    std::vector<int> freq_indices;
    std::vector<float> bands_per_channel_freq;
};

struct RoformerSegmentManifest {
    int depth = 12;
    int num_bands = 0;
    int dim = 256;
    int time_batch = 1;
    int freq_batch = 16;
    std::string attention_op = "manual";
    std::vector<int> dim_inputs;
};

enum class RoformerPrecisionPolicy {
    Uniform,
    MetalFast,
    MetalAutocast,
};

enum class RoformerSegmentCachePolicy {
    All,
    TransformersOnly,
    None,
};

RoformerPrecisionPolicy roformer_precision_policy_from_name(const std::string& name);
std::string roformer_precision_policy_name(RoformerPrecisionPolicy policy);
RoformerSegmentCachePolicy roformer_segment_cache_policy_from_name(const std::string& name);
std::string roformer_segment_cache_policy_name(RoformerSegmentCachePolicy policy);

struct RoformerSeparatorOptions {
    std::string segment_dir;
    std::string metadata_path;
    MNNBackend backend = MNNBackend::CPU;
    MNNPrecision precision = MNNPrecision::Auto;
    RoformerPrecisionPolicy precision_policy = RoformerPrecisionPolicy::MetalFast;
    RoformerSegmentCachePolicy segment_cache_policy = RoformerSegmentCachePolicy::TransformersOnly;
    int threads = 1;
};

RoformerMetadata load_roformer_metadata(const std::string& path);
RoformerSegmentManifest load_roformer_manifest(const std::string& segment_dir);

class RoformerSeparator {
public:
    explicit RoformerSeparator(RoformerSeparatorOptions options);
    ~RoformerSeparator();

    RoformerSeparator(const RoformerSeparator&) = delete;
    RoformerSeparator& operator=(const RoformerSeparator&) = delete;
    RoformerSeparator(RoformerSeparator&&) noexcept;
    RoformerSeparator& operator=(RoformerSeparator&&) noexcept;

    std::vector<AudioBuffer> separate(const AudioBuffer& audio);
    const RoformerMetadata& metadata() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mss_mnn
