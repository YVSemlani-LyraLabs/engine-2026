#pragma once

// In-memory sample buffers plus the on-disk write-out.
//
// Samples are serialized as flat binary of the trivially-copyable records
// (samples.h) behind a small header (magic, record size, count). The
// Python/PyTorch trainer reads them back as a numpy structured dtype
// (train/data.py); the record-size field guards both sides against silent
// struct-layout drift.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace pkrbot::engine {

constexpr int BUFFER_SIZE = 100000;

// File layout: three uint64 header fields (magic "PKRS", sizeof(record),
// record count) followed by `count` raw records.
constexpr uint64_t SAMPLE_FILE_MAGIC = 0x53524B50;  // "PKRS" little-endian

// Fixed-capacity reservoir of training samples (Algorithm R, per Deep CFR /
// Brown et al.): every sample ever pushed is equally likely to be in the
// buffer, so the buffer is an unbiased draw across all CFR iterations.
// Linear-CFR weighting is therefore applied at training time from each
// sample's `iteration` field (see samples.h), not by biasing retention here.
//
// push()/size() are thread-safe: the buffers are shared by the parallel
// traversal workers that feed the inference batcher (train.cpp). Reservoir
// uniformity is order-independent, so interleaved pushes from many workers
// stay an unbiased draw; only the exact retained set varies with interleaving.
template <typename T>
struct SampleBuffer {
  std::array<T, BUFFER_SIZE> samples;
  long long seen = 0;  // total samples pushed; only the first size() slots are valid

  // `seed` drives reservoir eviction only; it is deliberately separate from
  // the traversal rng so buffer bookkeeping never perturbs game trajectories.
  explicit SampleBuffer(uint64_t seed) : rng(seed) {}

  void push(const T& sample) {
    std::lock_guard<std::mutex> lock(mu);
    if (seen < BUFFER_SIZE) {
      samples[seen] = sample;
    } else {
      // Keep the new sample with probability BUFFER_SIZE / (seen + 1).
      long long j = std::uniform_int_distribution<long long>(0, seen)(rng);
      if (j < BUFFER_SIZE) samples[j] = sample;
    }
    ++seen;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu);
    return static_cast<size_t>(std::min<long long>(seen, BUFFER_SIZE));
  }

  long long totalSeen() const {
    std::lock_guard<std::mutex> lock(mu);
    return seen;
  }

 private:
  mutable std::mutex mu;
  std::mt19937_64 rng;
};

// Dump the retained samples to `path` (see the file-layout comment above).
// Call after the traversal workers have joined; takes the buffer lock only to
// read `seen`.
template <typename T>
void writeSamples(const SampleBuffer<T>& buffer, const std::string& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open sample file for writing: " + path);
  uint64_t header[3] = {SAMPLE_FILE_MAGIC, sizeof(T), buffer.size()};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(buffer.samples.data()),
            static_cast<std::streamsize>(buffer.size() * sizeof(T)));
  if (!out) throw std::runtime_error("short write to sample file: " + path);
}

}  // namespace pkrbot::engine
