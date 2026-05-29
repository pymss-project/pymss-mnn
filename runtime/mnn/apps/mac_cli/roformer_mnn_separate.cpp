#include "mss_mnn/roformer_separator.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string preset;
    std::string segment_dir;
    std::string core_model;
    std::string metadata;
    std::string input_wav;
    std::string output_dir;
    mss_mnn::MNNBackend backend = mss_mnn::MNNBackend::CPU;
    mss_mnn::MNNPrecision precision = mss_mnn::MNNPrecision::Auto;
    mss_mnn::RoformerPrecisionPolicy precision_policy = mss_mnn::RoformerPrecisionPolicy::MetalFast;
    mss_mnn::RoformerSegmentCachePolicy segment_cache_policy = mss_mnn::RoformerSegmentCachePolicy::TransformersOnly;
    int threads = 1;
    bool profile = false;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --preset bsr_hyperace_voc (--segments dir | --core-model model.mnn) --metadata file.json "
        << "--input input.wav --output-dir out [--backend cpu|auto|metal|opencl|vulkan] "
        << "[--precision auto|normal|high|low|low-bf16] "
        << "[--precision-policy uniform|metal-fast|metal-autocast] "
        << "[--segment-cache all|transformers|blocks|none] [--threads 1] [--profile]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + option);
            }
            return argv[++i];
        };
        if (key == "--preset") {
            args.preset = require_value(key);
        } else if (key == "--segments") {
            args.segment_dir = require_value(key);
        } else if (key == "--core-model") {
            args.core_model = require_value(key);
        } else if (key == "--metadata") {
            args.metadata = require_value(key);
        } else if (key == "--input") {
            args.input_wav = require_value(key);
        } else if (key == "--output-dir") {
            args.output_dir = require_value(key);
        } else if (key == "--backend") {
            args.backend = mss_mnn::mnn_backend_from_name(require_value(key));
        } else if (key == "--precision") {
            args.precision = mss_mnn::mnn_precision_from_name(require_value(key));
        } else if (key == "--precision-policy") {
            args.precision_policy = mss_mnn::roformer_precision_policy_from_name(require_value(key));
        } else if (key == "--segment-cache") {
            args.segment_cache_policy = mss_mnn::roformer_segment_cache_policy_from_name(require_value(key));
        } else if (key == "--threads") {
            args.threads = std::stoi(require_value(key));
        } else if (key == "--profile") {
            args.profile = true;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    if (args.preset.empty() || args.metadata.empty() || args.input_wav.empty() || args.output_dir.empty()) {
        throw std::runtime_error("missing required arguments");
    }
    if (args.segment_dir.empty() == args.core_model.empty()) {
        throw std::runtime_error("provide exactly one of --segments or --core-model");
    }
    return args;
}

std::string basename_no_ext(const std::string& path) {
    const auto stem = std::filesystem::path(path).stem().string();
    return stem.empty() ? "audio" : stem;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        mss_mnn::RoformerSeparatorOptions options;
        options.segment_dir = args.segment_dir;
        options.core_model_path = args.core_model;
        options.metadata_path = args.metadata;
        options.backend = args.backend;
        options.precision = args.precision;
        options.precision_policy = args.precision_policy;
        options.segment_cache_policy = args.segment_cache_policy;
        options.threads = args.threads;
        options.profile = args.profile;

        mss_mnn::RoformerSeparator separator(options);
        const auto input = mss_mnn::read_wav(args.input_wav);
        const auto outputs = separator.separate(input);

        std::filesystem::create_directories(args.output_dir);
        const auto& metadata = separator.metadata();
        const std::string stem_base = basename_no_ext(args.input_wav);
        for (std::size_t stem = 0; stem < outputs.size(); ++stem) {
            const std::string name = stem < metadata.source_names.size() ? metadata.source_names[stem] : std::to_string(stem);
            mss_mnn::write_wav_float32(args.output_dir + "/" + stem_base + "_" + name + ".wav", outputs[stem]);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        usage(argv[0]);
        return 1;
    }
}
