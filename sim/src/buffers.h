#pragma once

// In-memory sample buffers plus the write-out seam.
//
// On-disk plan (pass 2): SampleSink implementations serialize samples as
// chunked flat binary of the trivially-copyable records (samples.h) with a
// small header (magic, version, record size). The Python/PyTorch trainer
// reads the chunks back as a numpy structured dtype.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>

namespace pkrbot::engine {

constexpr int BUFFER_SIZE = 100000;

// Sink for flushing samples out of the simulator (e.g. to disk for the
// trainer). The default is a no-op; a real binary writer lands in pass 2.
template <typename T>
struct SampleSink {
  virtual ~SampleSink() = default;
  virtual void write(const T* /*samples*/, size_t /*n*/) {}
};

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

}  // namespace pkrbot::engine
