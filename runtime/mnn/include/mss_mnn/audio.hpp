#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mss_mnn {

struct AudioBuffer {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> data;  // channel-major: channels x frames

    std::size_t frames() const {
        return channels <= 0 ? 0 : data.size() / static_cast<std::size_t>(channels);
    }
};

AudioBuffer read_wav(const std::string& path);
void write_wav_float32(const std::string& path, const AudioBuffer& audio);

}  // namespace mss_mnn
