#include "mss_mnn/mnn_mask_core.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string model;
    std::string input;
    std::string output;
    std::string shape;
    std::string input_name = "stft_repr";
    std::string output_name = "mask";
    mss_mnn::MNNBackend backend = mss_mnn::MNNBackend::CPU;
    mss_mnn::MNNPrecision precision = mss_mnn::MNNPrecision::Auto;
    int threads = 1;
    int attention_option = 0;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model model.mnn --input input.f32 --shape 1,2050,938,2 --output out.f32\n"
        << "Optional: --input-name stft_repr --output-name mask --backend cpu|auto|metal|opencl|vulkan "
        << "--precision auto|normal|high|low|low-bf16 --attention-option 8 --threads 1\n";
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
        if (key == "--model") {
            args.model = require_value(key);
        } else if (key == "--input") {
            args.input = require_value(key);
        } else if (key == "--output") {
            args.output = require_value(key);
        } else if (key == "--shape") {
            args.shape = require_value(key);
        } else if (key == "--input-name") {
            args.input_name = require_value(key);
        } else if (key == "--output-name") {
            args.output_name = require_value(key);
        } else if (key == "--backend") {
            args.backend = mss_mnn::mnn_backend_from_name(require_value(key));
        } else if (key == "--precision") {
            args.precision = mss_mnn::mnn_precision_from_name(require_value(key));
        } else if (key == "--attention-option") {
            args.attention_option = std::stoi(require_value(key));
        } else if (key == "--threads") {
            args.threads = std::stoi(require_value(key));
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    if (args.model.empty() || args.input.empty() || args.output.empty() || args.shape.empty()) {
        throw std::runtime_error("missing required arguments");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        mss_mnn::MaskCoreOptions options;
        options.input_name = args.input_name;
        options.output_name = args.output_name;
        options.backend = args.backend;
        options.precision = args.precision;
        options.threads = args.threads;
        options.attention_option = args.attention_option;

        mss_mnn::MNNMaskCore core(args.model, options);
        const auto input = mss_mnn::read_f32_file(args.input);
        const auto output = core.run(input, mss_mnn::parse_shape(args.shape));
        mss_mnn::write_f32_file(args.output, output);

        std::cerr << "wrote " << output.size() << " float32 values";
        std::cerr << " shape=";
        const auto& shape = core.output_shape();
        for (std::size_t i = 0; i < shape.size(); ++i) {
            std::cerr << (i == 0 ? "" : ",") << shape[i];
        }
        std::cerr << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        usage(argv[0]);
        return 1;
    }
}
