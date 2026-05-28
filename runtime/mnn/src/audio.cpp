#include "mss_mnn/audio.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mss_mnn {
namespace {

std::uint32_t read_u32(std::istream& stream) {
    unsigned char b[4];
    stream.read(reinterpret_cast<char*>(b), 4);
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint16_t read_u16(std::istream& stream) {
    unsigned char b[2];
    stream.read(reinterpret_cast<char*>(b), 2);
    return static_cast<std::uint16_t>(b[0]) | (static_cast<std::uint16_t>(b[1]) << 8);
}

void write_u32(std::ostream& stream, std::uint32_t value) {
    const unsigned char b[4] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>((value >> 16) & 0xff),
        static_cast<unsigned char>((value >> 24) & 0xff),
    };
    stream.write(reinterpret_cast<const char*>(b), 4);
}

void write_u16(std::ostream& stream, std::uint16_t value) {
    const unsigned char b[2] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
    };
    stream.write(reinterpret_cast<const char*>(b), 2);
}

}  // namespace

AudioBuffer read_wav(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open wav: " + path);
    }
    char riff[4];
    stream.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        throw std::runtime_error("wav is missing RIFF header: " + path);
    }
    read_u32(stream);
    char wave[4];
    stream.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        throw std::runtime_error("wav is missing WAVE header: " + path);
    }

    int format = 0;
    int channels = 0;
    int sample_rate = 0;
    int bits_per_sample = 0;
    std::vector<unsigned char> data_bytes;
    while (stream) {
        char id[4];
        stream.read(id, 4);
        if (!stream) {
            break;
        }
        const auto size = read_u32(stream);
        const std::string chunk(id, id + 4);
        if (chunk == "fmt ") {
            format = read_u16(stream);
            channels = read_u16(stream);
            sample_rate = static_cast<int>(read_u32(stream));
            read_u32(stream);
            read_u16(stream);
            bits_per_sample = read_u16(stream);
            if (size > 16) {
                stream.seekg(size - 16, std::ios::cur);
            }
        } else if (chunk == "data") {
            data_bytes.resize(size);
            stream.read(reinterpret_cast<char*>(data_bytes.data()), size);
        } else {
            stream.seekg(size, std::ios::cur);
        }
        if (size % 2) {
            stream.seekg(1, std::ios::cur);
        }
    }
    if (channels <= 0 || sample_rate <= 0 || data_bytes.empty()) {
        throw std::runtime_error("wav is missing fmt/data chunks: " + path);
    }
    if (!((format == 3 && bits_per_sample == 32) || (format == 1 && bits_per_sample == 16))) {
        throw std::runtime_error("only float32 or pcm16 wav is supported: " + path);
    }
    const std::size_t bytes_per_sample = static_cast<std::size_t>(bits_per_sample / 8);
    const std::size_t frames = data_bytes.size() / (bytes_per_sample * static_cast<std::size_t>(channels));
    AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = channels;
    audio.data.assign(static_cast<std::size_t>(channels) * frames, 0.0f);
    const unsigned char* ptr = data_bytes.data();
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (int ch = 0; ch < channels; ++ch) {
            float value = 0.0f;
            if (format == 3) {
                std::uint32_t raw = static_cast<std::uint32_t>(ptr[0]) | (static_cast<std::uint32_t>(ptr[1]) << 8) |
                                    (static_cast<std::uint32_t>(ptr[2]) << 16) | (static_cast<std::uint32_t>(ptr[3]) << 24);
                std::memcpy(&value, &raw, sizeof(float));
            } else {
                const std::int16_t raw = static_cast<std::int16_t>(static_cast<std::uint16_t>(ptr[0]) | (static_cast<std::uint16_t>(ptr[1]) << 8));
                value = static_cast<float>(raw) / 32768.0f;
            }
            audio.data[static_cast<std::size_t>(ch) * frames + frame] = value;
            ptr += bytes_per_sample;
        }
    }
    return audio;
}

void write_wav_float32(const std::string& path, const AudioBuffer& audio) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open wav for write: " + path);
    }
    const std::uint32_t frames = static_cast<std::uint32_t>(audio.frames());
    const std::uint16_t channels = static_cast<std::uint16_t>(audio.channels);
    const std::uint32_t data_size = frames * channels * sizeof(float);
    const std::uint32_t fmt_size = 16;
    const std::uint32_t riff_size = 4 + (8 + fmt_size) + (8 + data_size);
    stream.write("RIFF", 4);
    write_u32(stream, riff_size);
    stream.write("WAVE", 4);
    stream.write("fmt ", 4);
    write_u32(stream, fmt_size);
    write_u16(stream, 3);
    write_u16(stream, channels);
    write_u32(stream, static_cast<std::uint32_t>(audio.sample_rate));
    write_u32(stream, static_cast<std::uint32_t>(audio.sample_rate) * channels * sizeof(float));
    write_u16(stream, static_cast<std::uint16_t>(channels * sizeof(float)));
    write_u16(stream, 32);
    stream.write("data", 4);
    write_u32(stream, data_size);
    for (std::size_t frame = 0; frame < audio.frames(); ++frame) {
        for (int ch = 0; ch < audio.channels; ++ch) {
            const float value = audio.data[static_cast<std::size_t>(ch) * audio.frames() + frame];
            stream.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }
    }
}

}  // namespace mss_mnn
