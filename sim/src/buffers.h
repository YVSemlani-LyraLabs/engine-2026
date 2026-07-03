#pragma once

// In-memory sample buffers plus the write-out seam.
//
// On-disk plan (pass 2): SampleSink implementations serialize samples as
// chunked flat binary of the trivially-copyable records (samples.h) with a
// small header (magic, version, record size). The Python/PyTorch trainer
// reads the chunks back as a numpy structured dtype.

#include <cstddef>

#include <array>

namespace pkrbot::engine {

constexpr int BUFFER_SIZE = 100000;

// Sink for flushing samples out of the simulator (e.g. to disk for the
// trainer). The default is a no-op; a real binary writer lands in pass 2.
template <typename T>
struct SampleSink {
  virtual ~SampleSink() = default;
  virtual void write(const T* /*samples*/, size_t /*n*/) {}
};

// Fixed-capacity buffer of training samples.
// TODO(pass 2): replace ring overwrite with reservoir sampling per
// Brown et al.; overwriting the oldest samples biases the buffer toward
// recent iterations.
template <typename T>
struct SampleBuffer {
  std::array<T, BUFFER_SIZE> samples;
  int idx = 0;

  void push(const T& sample) {
    samples[idx] = sample;
    idx = (idx + 1) % BUFFER_SIZE;
  }
};

}  // namespace pkrbot::engine
