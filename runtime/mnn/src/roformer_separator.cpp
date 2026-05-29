#include "mss_mnn/roformer_separator.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mss_mnn {

using Complex = std::complex<float>;

std::string read_text(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open text file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

std::size_t find_key(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto pos = text.find(needle);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing json key: " + key);
    }
    return pos + needle.size();
}

std::size_t find_value_start(const std::string& text, const std::string& key) {
    auto pos = find_key(text, key);
    pos = text.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing ':' for json key: " + key);
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

int json_int(const std::string& text, const std::string& key) {
    const auto pos = find_value_start(text, key);
    return std::stoi(text.substr(pos));
}

std::string json_string(const std::string& text, const std::string& key) {
    auto pos = find_value_start(text, key);
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("json key is not a string: " + key);
    }
    ++pos;
    const auto end = text.find('"', pos);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated json string: " + key);
    }
    return text.substr(pos, end - pos);
}

std::string json_string_or_default(const std::string& text, const std::string& key, const std::string& fallback) {
    const std::string needle = "\"" + key + "\"";
    if (text.find(needle) == std::string::npos) {
        return fallback;
    }
    return json_string(text, key);
}

std::string json_array_body(const std::string& text, const std::string& key) {
    auto pos = find_value_start(text, key);
    if (pos >= text.size() || text[pos] != '[') {
        throw std::runtime_error("json key is not an array: " + key);
    }
    int depth = 0;
    bool in_string = false;
    for (std::size_t i = pos; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"' && (i == 0 || text[i - 1] != '\\')) {
            in_string = !in_string;
        }
        if (in_string) {
            continue;
        }
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                return text.substr(pos + 1, i - pos - 1);
            }
        }
    }
    throw std::runtime_error("unterminated json array: " + key);
}

std::vector<int> json_int_array(const std::string& text, const std::string& key) {
    const auto body = json_array_body(text, key);
    std::vector<int> values;
    std::size_t pos = 0;
    while (pos < body.size()) {
        while (pos < body.size() && !(body[pos] == '-' || std::isdigit(static_cast<unsigned char>(body[pos])))) {
            ++pos;
        }
        if (pos >= body.size()) {
            break;
        }
        std::size_t end = pos + 1;
        while (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end]))) {
            ++end;
        }
        values.push_back(std::stoi(body.substr(pos, end - pos)));
        pos = end;
    }
    return values;
}

std::vector<float> json_float_array(const std::string& text, const std::string& key) {
    const auto body = json_array_body(text, key);
    std::vector<float> values;
    std::size_t pos = 0;
    while (pos < body.size()) {
        while (pos < body.size() && !(body[pos] == '-' || body[pos] == '.' || std::isdigit(static_cast<unsigned char>(body[pos])))) {
            ++pos;
        }
        if (pos >= body.size()) {
            break;
        }
        char* end_ptr = nullptr;
        const float value = std::strtof(body.c_str() + pos, &end_ptr);
        values.push_back(value);
        pos = static_cast<std::size_t>(end_ptr - body.c_str());
    }
    return values;
}

std::vector<std::string> json_string_array(const std::string& text, const std::string& key) {
    const auto body = json_array_body(text, key);
    std::vector<std::string> values;
    std::size_t pos = 0;
    while (true) {
        pos = body.find('"', pos);
        if (pos == std::string::npos) {
            break;
        }
        const auto end = body.find('"', pos + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("unterminated string array value: " + key);
        }
        values.push_back(body.substr(pos + 1, end - pos - 1));
        pos = end + 1;
    }
    return values;
}

RoformerMetadata load_roformer_metadata(const std::string& path) {
    const std::string text = read_text(path);
    RoformerMetadata metadata;
    metadata.preset = json_string(text, "preset");
    metadata.model_type = json_string(text, "model_type");
    metadata.sample_rate = json_int(text, "sample_rate");
    metadata.chunk_size = json_int(text, "chunk_size");
    metadata.overlap_size = json_int(text, "overlap_size");
    metadata.source_names = json_string_array(text, "source_names");
    metadata.stems = static_cast<int>(metadata.source_names.size());
    metadata.n_fft = json_int(text, "n_fft");
    metadata.hop_length = json_int(text, "hop_length");
    metadata.win_length = json_int(text, "win_length");
    const auto input_shape = json_int_array(text, "input_shape");
    const auto output_shape = json_int_array(text, "output_shape");
    if (input_shape.size() != 4 || output_shape.size() != 5) {
        throw std::runtime_error("unexpected metadata input/output shape");
    }
    metadata.input_freq_channels = input_shape[1];
    metadata.output_freq_channels = output_shape[2];
    metadata.full_freq_channels = (metadata.n_fft / 2 + 1) * 2;
    metadata.frames = input_shape[2];
    metadata.mask_mode = json_string_or_default(text, "mask_mode", "no_segm");
    if (metadata.model_type == "MelBandRoformer") {
        metadata.freq_indices = json_int_array(text, "freq_indices");
        metadata.bands_per_channel_freq = json_float_array(text, "num_bands_per_channel_freq");
    }
    return metadata;
}

RoformerSegmentManifest load_roformer_manifest(const std::string& segment_dir) {
    const std::string text = read_text(segment_dir + "/manifest.json");
    RoformerSegmentManifest manifest;
    manifest.preset = json_string_or_default(text, "preset", "");
    manifest.model_type = json_string_or_default(text, "model_type", "");
    manifest.depth = json_int(text, "depth");
    manifest.num_bands = json_int(text, "num_bands");
    manifest.time_batch = json_int(text, "time_batch");
    manifest.freq_batch = json_int(text, "freq_batch");
    manifest.attention_op = json_string_or_default(text, "attention_op", "manual");
    manifest.transformer_split = json_string_or_default(text, "transformer_split", "fused");
    manifest.dim_inputs = json_int_array(text, "dim_inputs");
    const auto band_shape = json_int_array(text, "band_shape");
    if (band_shape.size() != 4) {
        throw std::runtime_error("unexpected manifest band_shape");
    }
    manifest.dim = band_shape[3];
    return manifest;
}

RoformerPrecisionPolicy roformer_precision_policy_from_name(const std::string& name) {
    std::string value = name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "uniform") {
        return RoformerPrecisionPolicy::Uniform;
    }
    if (value == "metal-fast" || value == "fast" || value == "mobile-fast") {
        return RoformerPrecisionPolicy::MetalFast;
    }
    if (value == "metal-autocast" || value == "autocast" || value == "metal-balanced" || value == "balanced") {
        return RoformerPrecisionPolicy::MetalAutocast;
    }
    throw std::runtime_error("unknown RoFormer precision policy: " + name);
}

std::string roformer_precision_policy_name(RoformerPrecisionPolicy policy) {
    switch (policy) {
        case RoformerPrecisionPolicy::Uniform:
            return "uniform";
        case RoformerPrecisionPolicy::MetalFast:
            return "metal-fast";
        case RoformerPrecisionPolicy::MetalAutocast:
            return "metal-autocast";
    }
    return "uniform";
}

RoformerSegmentCachePolicy roformer_segment_cache_policy_from_name(const std::string& name) {
    std::string value = name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "all") {
        return RoformerSegmentCachePolicy::All;
    }
    if (value == "transformers" || value == "transformer" || value == "core") {
        return RoformerSegmentCachePolicy::TransformersOnly;
    }
    if (value == "none" || value == "off" || value == "no-cache") {
        return RoformerSegmentCachePolicy::None;
    }
    throw std::runtime_error("unknown RoFormer segment cache policy: " + name);
}

std::string roformer_segment_cache_policy_name(RoformerSegmentCachePolicy policy) {
    switch (policy) {
        case RoformerSegmentCachePolicy::All:
            return "all";
        case RoformerSegmentCachePolicy::TransformersOnly:
            return "transformers";
        case RoformerSegmentCachePolicy::None:
            return "none";
    }
    return "all";
}

std::vector<float> hann_window(int size) {
    std::vector<float> window(size);
    constexpr float pi = 3.14159265358979323846f;
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * pi * static_cast<float>(i) / static_cast<float>(size)));
    }
    return window;
}

int reflect_index(int index, int length) {
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - 2 - index;
        }
    }
    return index;
}

std::vector<float> reflect_pad_1d(const std::vector<float>& input, int channels, int left, int right) {
    const int length = static_cast<int>(input.size() / static_cast<std::size_t>(channels));
    std::vector<float> output(static_cast<std::size_t>(channels) * static_cast<std::size_t>(length + left + right));
    for (int ch = 0; ch < channels; ++ch) {
        for (int i = 0; i < length + left + right; ++i) {
            const int src = reflect_index(i - left, length);
            output[static_cast<std::size_t>(ch) * (length + left + right) + i] = input[static_cast<std::size_t>(ch) * length + src];
        }
    }
    return output;
}

std::vector<float> extract_chunk(const std::vector<float>& mix, int channels, int start, int chunk_size, int* valid_length) {
    const int total = static_cast<int>(mix.size() / static_cast<std::size_t>(channels));
    const int length = std::min(chunk_size, total - start);
    *valid_length = length;
    std::vector<float> chunk(static_cast<std::size_t>(channels) * chunk_size, 0.0f);
    for (int ch = 0; ch < channels; ++ch) {
        for (int i = 0; i < length; ++i) {
            chunk[static_cast<std::size_t>(ch) * chunk_size + i] = mix[static_cast<std::size_t>(ch) * total + start + i];
        }
        if (length < chunk_size && length > chunk_size / 2 + 1) {
            for (int i = length; i < chunk_size; ++i) {
                const int reflected = reflect_index(i, length);
                chunk[static_cast<std::size_t>(ch) * chunk_size + i] = chunk[static_cast<std::size_t>(ch) * chunk_size + reflected];
            }
        }
    }
    return chunk;
}

void fft(std::vector<Complex>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    constexpr float pi = 3.14159265358979323846f;
    for (int len = 2; len <= n; len <<= 1) {
        const float angle = 2.0f * pi / static_cast<float>(len) * (inverse ? 1.0f : -1.0f);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& value : a) {
            value /= static_cast<float>(n);
        }
    }
}

std::vector<float> stft_roformer(const std::vector<float>& chunk, int channels, const RoformerMetadata& metadata) {
    const int n_fft = metadata.n_fft;
    const int hop = metadata.hop_length;
    const int pad = n_fft / 2;
    const int length = static_cast<int>(chunk.size() / static_cast<std::size_t>(channels));
    const auto padded = reflect_pad_1d(chunk, channels, pad, pad);
    const int padded_length = length + 2 * pad;
    const int frames = (padded_length - n_fft) / hop + 1;
    const int freq_bins = n_fft / 2 + 1;
    if (frames != metadata.frames) {
        throw std::runtime_error("STFT frame count does not match metadata");
    }
    const auto window = hann_window(metadata.win_length);
    std::vector<float> out(static_cast<std::size_t>(freq_bins * channels) * frames * 2, 0.0f);
    std::vector<Complex> buffer(static_cast<std::size_t>(n_fft));
    for (int ch = 0; ch < channels; ++ch) {
        for (int frame = 0; frame < frames; ++frame) {
            std::fill(buffer.begin(), buffer.end(), Complex(0.0f, 0.0f));
            const int base = frame * hop;
            for (int i = 0; i < metadata.win_length; ++i) {
                buffer[i] = Complex(padded[static_cast<std::size_t>(ch) * padded_length + base + i] * window[i], 0.0f);
            }
            fft(buffer, false);
            for (int freq = 0; freq < freq_bins; ++freq) {
                const int fc = freq * channels + ch;
                const std::size_t idx = ((static_cast<std::size_t>(fc) * frames + frame) * 2);
                out[idx] = buffer[freq].real();
                out[idx + 1] = buffer[freq].imag();
            }
        }
    }
    return out;
}

std::vector<float> istft_roformer(const std::vector<float>& masked, int stems, int channels, int output_length, const RoformerMetadata& metadata) {
    const int n_fft = metadata.n_fft;
    const int hop = metadata.hop_length;
    const int pad = n_fft / 2;
    const int frames = metadata.frames;
    const int freq_bins = n_fft / 2 + 1;
    const int padded_length = output_length + 2 * pad;
    const auto window = hann_window(metadata.win_length);
    std::vector<float> recon(static_cast<std::size_t>(stems * channels) * padded_length, 0.0f);
    std::vector<float> envelope(static_cast<std::size_t>(padded_length), 0.0f);
    for (int frame = 0; frame < frames; ++frame) {
        const int base = frame * hop;
        for (int i = 0; i < metadata.win_length; ++i) {
            envelope[base + i] += window[i] * window[i];
        }
    }
    std::vector<Complex> spectrum(static_cast<std::size_t>(n_fft));
    for (int stem = 0; stem < stems; ++stem) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int frame = 0; frame < frames; ++frame) {
                std::fill(spectrum.begin(), spectrum.end(), Complex(0.0f, 0.0f));
                for (int freq = 0; freq < freq_bins; ++freq) {
                    const int fc = freq * channels + ch;
                    const std::size_t idx = ((((static_cast<std::size_t>(stem) * (freq_bins * channels) + fc) * frames + frame) * 2));
                    spectrum[freq] = Complex(masked[idx], masked[idx + 1]);
                }
                for (int freq = 1; freq < freq_bins - 1; ++freq) {
                    spectrum[n_fft - freq] = std::conj(spectrum[freq]);
                }
                fft(spectrum, true);
                const int base = frame * hop;
                float* dest = recon.data() + (static_cast<std::size_t>(stem) * channels + ch) * padded_length;
                for (int i = 0; i < metadata.win_length; ++i) {
                    dest[base + i] += spectrum[i].real() * window[i];
                }
            }
            float* dest = recon.data() + (static_cast<std::size_t>(stem) * channels + ch) * padded_length;
            for (int i = 0; i < padded_length; ++i) {
                if (envelope[i] > 1e-11f) {
                    dest[i] /= envelope[i];
                }
            }
        }
    }
    std::vector<float> output(static_cast<std::size_t>(stems * channels) * output_length, 0.0f);
    for (int stem = 0; stem < stems; ++stem) {
        for (int ch = 0; ch < channels; ++ch) {
            const float* src = recon.data() + (static_cast<std::size_t>(stem) * channels + ch) * padded_length + pad;
            float* dst = output.data() + (static_cast<std::size_t>(stem) * channels + ch) * output_length;
            std::copy(src, src + output_length, dst);
        }
    }
    return output;
}

std::vector<float> windowing_array(int chunk_size, int fade_size) {
    std::vector<float> window(static_cast<std::size_t>(chunk_size), 1.0f);
    if (fade_size <= 0) {
        return window;
    }
    for (int i = 0; i < fade_size; ++i) {
        window[i] *= static_cast<float>(i) / static_cast<float>(fade_size - 1);
        window[chunk_size - fade_size + i] *= 1.0f - static_cast<float>(i) / static_cast<float>(fade_size - 1);
    }
    return window;
}

std::vector<float> chunk_window(int start, int total_length, int chunk_size, int fade_size, const std::vector<float>& normal) {
    const int length = std::min(chunk_size, total_length - start);
    std::vector<float> window = normal;
    if (start == 0) {
        for (int i = 0; i < fade_size && i < chunk_size; ++i) {
            window[i] = 1.0f;
        }
    }
    if (start + length >= total_length) {
        const int begin = std::max(0, length - fade_size);
        for (int i = begin; i < length; ++i) {
            window[i] = 1.0f;
        }
    }
    return window;
}

class SegmentRuntime {
public:
    SegmentRuntime(std::string segment_dir,
                   const RoformerSegmentManifest& manifest,
                   int stems,
                   std::string mask_mode,
                   MNNBackend backend,
                   MNNPrecision precision,
                   RoformerPrecisionPolicy precision_policy,
                   RoformerSegmentCachePolicy segment_cache_policy,
                   int threads)
        : segment_dir_(std::move(segment_dir)),
          manifest_(manifest),
          stems_(stems),
          mask_mode_(std::move(mask_mode)),
          backend_(backend),
          precision_(precision),
          precision_policy_(precision_policy),
          segment_cache_policy_(segment_cache_policy),
          threads_(threads) {}

    std::vector<float> run(const std::string& name, const std::vector<float>& input, const std::vector<int>& shape) {
        if (!should_cache(name)) {
            return create_runner(name).run(input, shape);
        }
        auto it = runners_.find(name);
        if (it == runners_.end()) {
            it = runners_.emplace(name, create_runner(name)).first;
        }
        return it->second.run(input, shape);
    }

    std::vector<float> operator()(const std::vector<float>& stft_repr, int freq_channels, int frames) {
        std::vector<float> x = run("band_split", stft_repr, {1, freq_channels, frames, 2});
        const int bands = manifest_.num_bands;
        const int dim = manifest_.dim;
        for (int layer = 0; layer < manifest_.depth; ++layer) {
            std::vector<float> next(x.size());
            for (int band = 0; band < bands; ++band) {
                std::vector<float> in(static_cast<std::size_t>(frames * dim));
                for (int t = 0; t < frames; ++t) {
                    const std::size_t src = ((static_cast<std::size_t>(t) * bands + band) * dim);
                    std::copy(x.begin() + static_cast<std::ptrdiff_t>(src), x.begin() + static_cast<std::ptrdiff_t>(src + dim), in.begin() + static_cast<std::ptrdiff_t>(t * dim));
                }
                auto out = run_transformer("layer_" + two(layer) + "_time", in, {1, frames, dim});
                for (int t = 0; t < frames; ++t) {
                    const std::size_t dst = ((static_cast<std::size_t>(t) * bands + band) * dim);
                    std::copy(out.begin() + static_cast<std::ptrdiff_t>(t * dim), out.begin() + static_cast<std::ptrdiff_t>((t + 1) * dim), next.begin() + static_cast<std::ptrdiff_t>(dst));
                }
            }
            x.swap(next);
            next.assign(x.size(), 0.0f);
            for (int start = 0; start < frames; start += manifest_.freq_batch) {
                const int actual = std::min(manifest_.freq_batch, frames - start);
                std::vector<float> in(static_cast<std::size_t>(manifest_.freq_batch * bands * dim), 0.0f);
                for (int t = 0; t < actual; ++t) {
                    const std::size_t src = (static_cast<std::size_t>(start + t) * bands * dim);
                    const std::size_t dst = static_cast<std::size_t>(t) * bands * dim;
                    std::copy(x.begin() + static_cast<std::ptrdiff_t>(src), x.begin() + static_cast<std::ptrdiff_t>(src + bands * dim), in.begin() + static_cast<std::ptrdiff_t>(dst));
                }
                auto out = run_transformer("layer_" + two(layer) + "_freq", in, {manifest_.freq_batch, bands, dim});
                for (int t = 0; t < actual; ++t) {
                    const std::size_t src = static_cast<std::size_t>(t) * bands * dim;
                    const std::size_t dst = static_cast<std::size_t>(start + t) * bands * dim;
                    std::copy(out.begin() + static_cast<std::ptrdiff_t>(src), out.begin() + static_cast<std::ptrdiff_t>(src + bands * dim), next.begin() + static_cast<std::ptrdiff_t>(dst));
                }
            }
            x.swap(next);
        }

        const int flat_dim = std::accumulate(manifest_.dim_inputs.begin(), manifest_.dim_inputs.end(), 0);
        const int stems = stems_;
        std::vector<float> mask(static_cast<std::size_t>(stems * (flat_dim / 2) * frames * 2), 0.0f);
        if (mask_mode_ != "segm_only") {
            int offset = 0;
            for (int band = 0; band < bands; ++band) {
                std::vector<float> in(static_cast<std::size_t>(frames * dim));
                for (int t = 0; t < frames; ++t) {
                    const std::size_t src = ((static_cast<std::size_t>(t) * bands + band) * dim);
                    std::copy(x.begin() + static_cast<std::ptrdiff_t>(src), x.begin() + static_cast<std::ptrdiff_t>(src + dim), in.begin() + static_cast<std::ptrdiff_t>(t * dim));
                }
                auto out = run("mask_band_" + two(band), in, {1, frames, dim});
                const int dim_in = manifest_.dim_inputs[band];
                const int out_stems = static_cast<int>(out.size() / static_cast<std::size_t>(frames * dim_in));
                for (int stem = 0; stem < out_stems; ++stem) {
                    for (int t = 0; t < frames; ++t) {
                        for (int d = 0; d < dim_in; ++d) {
                            const int selected = (offset + d) / 2;
                            const int ri = (offset + d) % 2;
                            const std::size_t src = (static_cast<std::size_t>(stem) * frames * dim_in + static_cast<std::size_t>(t) * dim_in + d);
                            const std::size_t dst = (((static_cast<std::size_t>(stem) * (flat_dim / 2) + selected) * frames + t) * 2 + ri);
                            mask[dst] = out[src];
                        }
                    }
                }
                offset += dim_in;
            }
        }
        if (mask_mode_ != "no_segm") {
            for (int stem = 0; stem < stems; ++stem) {
                auto out = run("segm_" + two(stem), x, {1, frames, bands, dim});
                if (static_cast<int>(out.size()) != frames * flat_dim) {
                    throw std::runtime_error("segm output size does not match expected flat mask size");
                }
                for (int t = 0; t < frames; ++t) {
                    for (int d = 0; d < flat_dim; ++d) {
                        const int selected = d / 2;
                        const int ri = d % 2;
                        const std::size_t src = static_cast<std::size_t>(t) * flat_dim + d;
                        const std::size_t dst = (((static_cast<std::size_t>(stem) * (flat_dim / 2) + selected) * frames + t) * 2 + ri);
                        if (mask_mode_ == "segm_only") {
                            mask[dst] = out[src];
                        } else {
                            mask[dst] += out[src];
                        }
                    }
                }
            }
        }
        return mask;
    }

private:
    static std::string two(int value) {
        return value < 10 ? "0" + std::to_string(value) : std::to_string(value);
    }

    static bool has_prefix(const std::string& value, const std::string& prefix) {
        return value.rfind(prefix, 0) == 0;
    }

    static bool has_suffix(const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    bool supports_split_attention_fp16() const {
        return manifest_.model_type.find("BSRoformer") != std::string::npos;
    }

    mss_mnn::MNNMaskCore create_runner(const std::string& name) const {
        mss_mnn::MaskCoreOptions options;
        options.input_name = "input";
        options.output_name = "output";
        options.backend = backend_;
        options.precision = segment_precision(name);
        options.threads = threads_;
        options.attention_option = segment_attention_option(name);
        return mss_mnn::MNNMaskCore(segment_dir_ + "/" + name + ".mnn", options);
    }

    std::vector<float> run_transformer(const std::string& name, const std::vector<float>& input, const std::vector<int>& shape) {
        if (manifest_.transformer_split == "attention_ffn") {
            return run(name + "_ffn", run(name + "_attn", input, shape), shape);
        }
        return run(name, input, shape);
    }

    bool should_cache(const std::string& name) const {
        switch (segment_cache_policy_) {
            case RoformerSegmentCachePolicy::All:
                return true;
            case RoformerSegmentCachePolicy::TransformersOnly:
                return name == "band_split" || has_prefix(name, "layer_");
            case RoformerSegmentCachePolicy::None:
                return false;
        }
        return true;
    }

    MNNPrecision segment_precision(const std::string& name) const {
        const bool metal_capable_backend = backend_ == MNNBackend::Metal || backend_ == MNNBackend::Auto;
        if (metal_capable_backend && manifest_.attention_op == "mnn" && manifest_.transformer_split == "attention_ffn") {
            if (has_suffix(name, "_attn")) {
                const bool force_high = precision_ == MNNPrecision::High || !supports_split_attention_fp16();
                return force_high ? MNNPrecision::High : MNNPrecision::Normal;
            }
            if (has_suffix(name, "_ffn")) {
                return MNNPrecision::High;
            }
        }
        if (metal_capable_backend && manifest_.attention_op == "mnn" && has_prefix(name, "layer_")) {
            // Unsplit transformer segments accumulate too much FP16 error on Metal.
            return MNNPrecision::High;
        }
        const bool uses_segment_autocast = precision_policy_ == RoformerPrecisionPolicy::MetalFast ||
                                           precision_policy_ == RoformerPrecisionPolicy::MetalAutocast;
        if (!uses_segment_autocast || !metal_capable_backend || precision_ != MNNPrecision::Auto) {
            return precision_;
        }
        if (name == "band_split") {
            return MNNPrecision::High;
        }
        if (precision_policy_ == RoformerPrecisionPolicy::MetalAutocast && has_prefix(name, "mask_")) {
            return MNNPrecision::High;
        }
        return MNNPrecision::Normal;
    }

    int segment_attention_option(const std::string& name) const {
        if (manifest_.attention_op != "mnn" || !has_prefix(name, "layer_") || has_suffix(name, "_ffn")) {
            return 0;
        }
        return 8;
    }

    std::string segment_dir_;
    RoformerSegmentManifest manifest_;
    int stems_;
    std::string mask_mode_;
    MNNBackend backend_;
    MNNPrecision precision_;
    RoformerPrecisionPolicy precision_policy_;
    RoformerSegmentCachePolicy segment_cache_policy_;
    int threads_;
    std::unordered_map<std::string, mss_mnn::MNNMaskCore> runners_;
};

std::vector<float> select_mbr_freqs(const std::vector<float>& stft, const RoformerMetadata& metadata) {
    std::vector<float> selected(static_cast<std::size_t>(metadata.freq_indices.size()) * metadata.frames * 2);
    for (std::size_t i = 0; i < metadata.freq_indices.size(); ++i) {
        const int src_fc = metadata.freq_indices[i];
        const std::size_t src = static_cast<std::size_t>(src_fc) * metadata.frames * 2;
        const std::size_t dst = i * metadata.frames * 2;
        std::copy(stft.begin() + static_cast<std::ptrdiff_t>(src), stft.begin() + static_cast<std::ptrdiff_t>(src + metadata.frames * 2), selected.begin() + static_cast<std::ptrdiff_t>(dst));
    }
    return selected;
}

std::vector<float> apply_mask(const std::vector<float>& stft, const std::vector<float>& mask, const RoformerMetadata& metadata) {
    const int frames = metadata.frames;
    const int full_fc = metadata.full_freq_channels;
    const int stems = metadata.stems;
    std::vector<float> out(static_cast<std::size_t>(stems * full_fc) * frames * 2, 0.0f);
    if (metadata.model_type == "MelBandRoformer") {
        std::vector<float> summed(static_cast<std::size_t>(stems * full_fc) * frames * 2, 0.0f);
        const int selected_fc = static_cast<int>(metadata.freq_indices.size());
        for (int stem = 0; stem < stems; ++stem) {
            for (int i = 0; i < selected_fc; ++i) {
                const int dst_fc = metadata.freq_indices[static_cast<std::size_t>(i)];
                for (int t = 0; t < frames; ++t) {
                    for (int ri = 0; ri < 2; ++ri) {
                        const std::size_t src = (((static_cast<std::size_t>(stem) * selected_fc + i) * frames + t) * 2 + ri);
                        const std::size_t dst = (((static_cast<std::size_t>(stem) * full_fc + dst_fc) * frames + t) * 2 + ri);
                        summed[dst] += mask[src];
                    }
                }
            }
        }
        for (int stem = 0; stem < stems; ++stem) {
            for (int fc = 0; fc < full_fc; ++fc) {
                const float denom = std::max(1e-8f, metadata.bands_per_channel_freq[static_cast<std::size_t>(fc)]);
                for (int t = 0; t < frames; ++t) {
                    const std::size_t stft_idx = (static_cast<std::size_t>(fc) * frames + t) * 2;
                    const std::size_t mask_idx = (((static_cast<std::size_t>(stem) * full_fc + fc) * frames + t) * 2);
                    const float ar = stft[stft_idx];
                    const float ai = stft[stft_idx + 1];
                    const float br = summed[mask_idx] / denom;
                    const float bi = summed[mask_idx + 1] / denom;
                    out[mask_idx] = ar * br - ai * bi;
                    out[mask_idx + 1] = ar * bi + ai * br;
                }
            }
        }
        return out;
    }

    for (int stem = 0; stem < stems; ++stem) {
        for (int fc = 0; fc < full_fc; ++fc) {
            for (int t = 0; t < frames; ++t) {
                const std::size_t a = (static_cast<std::size_t>(fc) * frames + t) * 2;
                const std::size_t b = (((static_cast<std::size_t>(stem) * full_fc + fc) * frames + t) * 2);
                const float ar = stft[a];
                const float ai = stft[a + 1];
                const float br = mask[b];
                const float bi = mask[b + 1];
                out[b] = ar * br - ai * bi;
                out[b + 1] = ar * bi + ai * br;
            }
        }
    }
    return out;
}

std::vector<float> separate_chunk(const std::vector<float>& chunk, int channels, const RoformerMetadata& metadata, SegmentRuntime& runtime) {
    auto stft = stft_roformer(chunk, channels, metadata);
    const std::vector<float> mask_input = metadata.model_type == "MelBandRoformer" ? select_mbr_freqs(stft, metadata) : stft;
    const int mask_freq_channels = metadata.model_type == "MelBandRoformer" ? static_cast<int>(metadata.freq_indices.size()) : metadata.output_freq_channels;
    auto mask = runtime(mask_input, mask_freq_channels, metadata.frames);
    auto masked = apply_mask(stft, mask, metadata);
    return istft_roformer(masked, metadata.stems, channels, static_cast<int>(chunk.size() / static_cast<std::size_t>(channels)), metadata);
}

struct RoformerSeparator::Impl {
    RoformerSeparatorOptions options;
    RoformerMetadata metadata;
    RoformerSegmentManifest manifest;
    SegmentRuntime runtime;

    explicit Impl(RoformerSeparatorOptions opts)
        : options(std::move(opts)),
          metadata(load_roformer_metadata(options.metadata_path)),
          manifest(load_roformer_manifest(options.segment_dir)),
          runtime(options.segment_dir,
                  manifest,
                  metadata.stems,
                  metadata.mask_mode,
                  options.backend,
                  options.precision,
                  options.precision_policy,
                  options.segment_cache_policy,
                  options.threads) {}
};

RoformerSeparator::RoformerSeparator(RoformerSeparatorOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

RoformerSeparator::~RoformerSeparator() = default;
RoformerSeparator::RoformerSeparator(RoformerSeparator&&) noexcept = default;
RoformerSeparator& RoformerSeparator::operator=(RoformerSeparator&&) noexcept = default;

std::vector<AudioBuffer> RoformerSeparator::separate(const AudioBuffer& audio) {
    const auto& metadata = impl_->metadata;
    if (audio.sample_rate != metadata.sample_rate) {
        throw std::runtime_error("input sample rate does not match metadata");
    }
    if (audio.channels != 2) {
        throw std::runtime_error("only stereo input is supported");
    }
    if (metadata.chunk_size <= metadata.overlap_size) {
        throw std::runtime_error("metadata chunk_size must be larger than overlap_size");
    }

    const int length_init = static_cast<int>(audio.frames());
    const int step = metadata.chunk_size - metadata.overlap_size;
    const int border = metadata.overlap_size;
    const int fade_size = std::min(metadata.chunk_size / 10, border);
    std::vector<float> mix = audio.data;
    if (length_init > 2 * border && border > 0) {
        mix = reflect_pad_1d(mix, audio.channels, border, border);
    }
    const int total = static_cast<int>(mix.size() / static_cast<std::size_t>(audio.channels));
    std::vector<int> starts;
    for (int start = 0; start < total; start += step) {
        starts.push_back(start);
    }

    const auto normal_window = windowing_array(metadata.chunk_size, fade_size);
    std::vector<float> result(static_cast<std::size_t>(metadata.stems * audio.channels) * total, 0.0f);
    std::vector<float> counter(static_cast<std::size_t>(total), 0.0f);
    for (int start : starts) {
        int valid = 0;
        auto chunk = extract_chunk(mix, audio.channels, start, metadata.chunk_size, &valid);
        auto separated = separate_chunk(chunk, audio.channels, metadata, impl_->runtime);
        auto window = chunk_window(start, total, metadata.chunk_size, fade_size, normal_window);
        for (int stem = 0; stem < metadata.stems; ++stem) {
            for (int ch = 0; ch < audio.channels; ++ch) {
                for (int i = 0; i < valid; ++i) {
                    const std::size_t dst = (static_cast<std::size_t>(stem * audio.channels + ch) * total + start + i);
                    const std::size_t src = (static_cast<std::size_t>(stem * audio.channels + ch) * metadata.chunk_size + i);
                    result[dst] += separated[src] * window[static_cast<std::size_t>(i)];
                }
            }
        }
        for (int i = 0; i < valid; ++i) {
            counter[static_cast<std::size_t>(start + i)] += window[static_cast<std::size_t>(i)];
        }
    }

    int crop_start = 0;
    int crop_end = total;
    if (length_init > 2 * border && border > 0) {
        crop_start = border;
        crop_end = border + length_init;
    }
    const int out_frames = crop_end - crop_start;
    std::vector<AudioBuffer> outputs(static_cast<std::size_t>(metadata.stems));
    for (int stem = 0; stem < metadata.stems; ++stem) {
        auto& out = outputs[static_cast<std::size_t>(stem)];
        out.sample_rate = metadata.sample_rate;
        out.channels = audio.channels;
        out.data.assign(static_cast<std::size_t>(audio.channels) * out_frames, 0.0f);
        for (int ch = 0; ch < audio.channels; ++ch) {
            for (int i = 0; i < out_frames; ++i) {
                const int pos = crop_start + i;
                const float denom = std::abs(counter[static_cast<std::size_t>(pos)]) > 1e-12f ? counter[static_cast<std::size_t>(pos)] : 1.0f;
                const std::size_t src = (static_cast<std::size_t>(stem * audio.channels + ch) * total + pos);
                out.data[static_cast<std::size_t>(ch) * out_frames + i] = result[src] / denom;
            }
        }
    }
    return outputs;
}

const RoformerMetadata& RoformerSeparator::metadata() const {
    return impl_->metadata;
}

}  // namespace mss_mnn
