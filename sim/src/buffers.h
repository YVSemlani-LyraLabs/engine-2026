#pragma once

// In-memory sample buffers plus the on-disk reservoir state.
//
// Samples are serialized as flat binary of the trivially-copyable records
// (samples.h) behind a small header (magic, version, record size, count,
// seen). The Python/PyTorch trainer reads them back as a numpy structured
// dtype (train/data.py); the record-size field guards both sides against
// silent struct-layout drift. Because the header carries `seen` alongside the
// retained records, a dump is the complete reservoir state: loadSamples() can
// restore it and keep pushing as if the buffer had never left memory, which
// is what makes the reservoir persistent across CFR iterations.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pkrbot::engine {

constexpr size_t DEFAULT_BUFFER_SIZE = 100000;

// File layout: five uint64 header fields (magic "PKRS", format version,
// sizeof(record), retained record count, total samples seen) followed by
// `count` raw records. Version 1 files had a three-field header (no version,
// no seen) and are rejected by both readers; regenerate them.
constexpr uint64_t SAMPLE_FILE_MAGIC = 0x53524B50;  // "PKRS" little-endian
constexpr uint64_t SAMPLE_FILE_VERSION = 2;

// Fixed-capacity reservoir of training samples (Algorithm R, per Deep CFR /
// Brown et al.): every sample ever pushed is equally likely to be in the
// buffer, so the buffer is an unbiased draw across all CFR iterations.
// Linear-CFR weighting is therefore applied at training time from each
// sample's `iteration` field (see samples.h), not by biasing retention here.
//
// pushBatch()/size() are thread-safe, but workers should not call pushBatch
// per sample: they write through a worker-local ReservoirWriter (below),
// which stages samples and folds them in a batch at a time so the lock is
// taken once per ~thousand samples instead of once per sample. Reservoir
// uniformity is order-independent, so interleaved batches from many workers
// stay an unbiased draw; only the exact retained set varies with
// interleaving.
template <typename T>
struct SampleBuffer {
  std::vector<T> samples;  // retained records; size() == min(seen, capacity)
  long long seen = 0;      // total samples ever pushed

  // `seed` drives reservoir eviction only; it is deliberately separate from
  // the traversal rng so buffer bookkeeping never perturbs game trajectories.
  SampleBuffer(size_t capacity, uint64_t seed) : capacity_(capacity), rng_(seed) {
    samples.reserve(capacity);
  }

  // Fold `n` samples into the reservoir under a single lock acquisition.
  // Copies happen only for accepted samples, and once seen >> capacity almost
  // every sample is rejected after one rng draw, so the critical section
  // stays small even for large batches.
  void pushBatch(const T* batch, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < n; ++i) {
      if (samples.size() < capacity_) {
        samples.push_back(batch[i]);
      } else {
        // Keep the new sample with probability capacity / (seen + 1).
        long long j = std::uniform_int_distribution<long long>(0, seen)(rng_);
        if (j < static_cast<long long>(capacity_)) samples[static_cast<size_t>(j)] = batch[i];
      }
      ++seen;
    }
  }

  void push(const T& sample) { pushBatch(&sample, 1); }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return samples.size();
  }

  long long totalSeen() const {
    std::lock_guard<std::mutex> lock(mu_);
    return seen;
  }

  size_t capacity() const { return capacity_; }

 private:
  size_t capacity_;
  mutable std::mutex mu_;
  std::mt19937_64 rng_;
};

// Worker-local staging front-end for a shared SampleBuffer. push() appends to
// a private array; when it fills, the whole batch is folded into the shared
// reservoir under one lock. This is what the traversal workers write through
// so they never contend on the buffer mutex per sample. Not thread-safe:
// one writer per worker.
//
// flush() must be called (or the writer destroyed) before reading the shared
// buffer's final contents.
template <typename T>
class ReservoirWriter {
 public:
  explicit ReservoirWriter(SampleBuffer<T>& buffer, size_t stagingCapacity = 1024)
      : buffer_(buffer), stagingCapacity_(stagingCapacity) {
    staging_.reserve(stagingCapacity_);
  }
  ReservoirWriter(const ReservoirWriter&) = delete;
  ReservoirWriter& operator=(const ReservoirWriter&) = delete;
  ~ReservoirWriter() { flush(); }

  void push(const T& sample) {
    staging_.push_back(sample);
    if (staging_.size() >= stagingCapacity_) flush();
  }

  void flush() {
    if (staging_.empty()) return;
    buffer_.pushBatch(staging_.data(), staging_.size());
    staging_.clear();
  }

 private:
  SampleBuffer<T>& buffer_;
  size_t stagingCapacity_;
  std::vector<T> staging_;
};

// Dump the full reservoir state to `path` (see the file-layout comment
// above). Call after the traversal workers have joined.
template <typename T>
void writeSamples(const SampleBuffer<T>& buffer, const std::string& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open sample file for writing: " + path);
  uint64_t header[5] = {SAMPLE_FILE_MAGIC, SAMPLE_FILE_VERSION, sizeof(T), buffer.size(),
                        static_cast<uint64_t>(buffer.totalSeen())};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(buffer.samples.data()),
            static_cast<std::streamsize>(buffer.size() * sizeof(T)));
  if (!out) throw std::runtime_error("short write to sample file: " + path);
}

// Restore reservoir state written by writeSamples() into an empty buffer, so
// pushes continue the same unbiased draw across iterations. Call before the
// traversal workers start.
//
// A saturated reservoir (seen > count) cannot be resumed into a different
// capacity: the retained set is a fixed-size sample of the past stream, so
// growing it would need already-discarded samples and shrinking it here would
// silently drop data — change capacity only between flywheel runs by
// re-sampling, or keep it fixed.
template <typename T>
void loadSamples(SampleBuffer<T>& buffer, const std::string& path) {
  if (buffer.totalSeen() != 0) {
    throw std::logic_error("loadSamples requires an empty buffer: " + path);
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open sample file: " + path);
  uint64_t header[5];
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!in) throw std::runtime_error("truncated sample file header: " + path);
  if (header[0] != SAMPLE_FILE_MAGIC) throw std::runtime_error("bad magic in " + path);
  if (header[1] != SAMPLE_FILE_VERSION) {
    throw std::runtime_error("unsupported sample file version in " + path +
                             " (want " + std::to_string(SAMPLE_FILE_VERSION) +
                             "; version-1 dumps predate the resumable header — regenerate)");
  }
  if (header[2] != sizeof(T)) {
    throw std::runtime_error("record size mismatch in " + path + ": file has " +
                             std::to_string(header[2]) + ", struct is " +
                             std::to_string(sizeof(T)));
  }
  uint64_t count = header[3];
  uint64_t seen = header[4];
  if (seen < count) throw std::runtime_error("corrupt sample file (seen < count): " + path);
  if (count > buffer.capacity()) {
    throw std::runtime_error("sample file " + path + " holds " + std::to_string(count) +
                             " records but buffer capacity is " +
                             std::to_string(buffer.capacity()));
  }
  if (seen > count && count != buffer.capacity()) {
    throw std::runtime_error("cannot resume saturated reservoir " + path + " (count " +
                             std::to_string(count) + ", seen " + std::to_string(seen) +
                             ") into capacity " + std::to_string(buffer.capacity()) +
                             "; capacity must match the file's retained count");
  }
  buffer.samples.resize(count);
  in.read(reinterpret_cast<char*>(buffer.samples.data()),
          static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error("truncated sample file: " + path);
  buffer.seen = static_cast<long long>(seen);
}

}  // namespace pkrbot::engine
