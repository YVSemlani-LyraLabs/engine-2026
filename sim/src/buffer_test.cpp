// Tests for the reservoir sample buffers (buffers.h): retention uniformity
// under multi-threaded batched writes, staging flush semantics, and the
// save/load reservoir-state roundtrip. Build target: buffer_test. Exits 0 and
// prints "all tests passed" on success.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "buffers.h"

// Not <cassert>'s assert: that is compiled out by -DNDEBUG in Release builds,
// which is how this target is normally built.
#define REQUIRE(cond)                                                                       \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::cerr << "REQUIRE failed at " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

namespace pkrbot::engine {
namespace {

struct TestRecord {
  int64_t id;
};

std::string tempPath(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("pkrbot_buffer_test_" + name)).string();
}

// Retention frequencies over many trials must be uniform across the stream —
// Algorithm R's guarantee — even when pushes arrive as interleaved batches
// from several ReservoirWriters with small staging arrays (partial flushes,
// racing pushBatch calls).
void testUniformRetentionUnderConcurrency() {
  constexpr int kTrials = 3000;
  constexpr size_t kCapacity = 32;
  constexpr int kStreamSize = 256;
  constexpr int kThreads = 4;
  constexpr size_t kStaging = 7;  // deliberately small and non-divisor: many partial batches

  std::vector<long long> retained(kStreamSize, 0);
  for (int trial = 0; trial < kTrials; ++trial) {
    SampleBuffer<TestRecord> buffer(kCapacity, static_cast<uint64_t>(trial) * 2654435761ULL + 1);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&buffer, t] {
        ReservoirWriter<TestRecord> writer(buffer, kStaging);
        int64_t lo = t * (kStreamSize / kThreads);
        int64_t hi = lo + kStreamSize / kThreads;
        for (int64_t id = lo; id < hi; ++id) writer.push({id});
      });
    }
    for (auto& t : threads) t.join();

    REQUIRE(buffer.size() == kCapacity);
    REQUIRE(buffer.totalSeen() == kStreamSize);
    for (const TestRecord& r : buffer.samples) ++retained[static_cast<size_t>(r.id)];
  }

  // Each id is retained with probability capacity/stream = 1/8 per trial.
  // Expected count 375, sigma ~18; +-100 is ~5.5 sigma, so a false failure is
  // vanishingly unlikely while a biased region (e.g. a staging batch that is
  // always kept or always dropped) lands far outside the band.
  const double expected = static_cast<double>(kTrials) * kCapacity / kStreamSize;
  for (int id = 0; id < kStreamSize; ++id) {
    REQUIRE(retained[id] > expected - 100 && retained[id] < expected + 100);
  }
}

// The writer must hold samples privately until its staging array fills, then
// hand them to the shared buffer in one batch; flush() and the destructor
// must deliver partial batches.
void testStagingFlush() {
  SampleBuffer<TestRecord> buffer(100, 42);
  {
    ReservoirWriter<TestRecord> writer(buffer, 10);
    for (int64_t i = 0; i < 9; ++i) writer.push({i});
    REQUIRE(buffer.totalSeen() == 0);  // staged, not yet visible
    writer.push({9});
    REQUIRE(buffer.totalSeen() == 10);  // staging filled -> one batch
    writer.push({10});
    REQUIRE(buffer.totalSeen() == 10);
    writer.flush();
    REQUIRE(buffer.totalSeen() == 11);
    writer.push({11});
  }  // destructor flushes the remainder
  REQUIRE(buffer.totalSeen() == 12);
  REQUIRE(buffer.size() == 12);
}

// A dump must be complete reservoir state: load restores samples and seen
// exactly, and continuing to push behaves as if the buffer never left memory.
void testSaveLoadRoundtrip() {
  const std::string path = tempPath("roundtrip.bin");

  SampleBuffer<TestRecord> original(16, 7);
  for (int64_t i = 0; i < 100; ++i) original.push({i});  // saturated: seen > capacity
  writeSamples(original, path);

  SampleBuffer<TestRecord> resumed(16, 8);
  loadSamples(resumed, path);
  REQUIRE(resumed.totalSeen() == 100);
  REQUIRE(resumed.size() == 16);
  for (size_t i = 0; i < 16; ++i) REQUIRE(resumed.samples[i].id == original.samples[i].id);

  resumed.push({100});
  REQUIRE(resumed.totalSeen() == 101);
  REQUIRE(resumed.size() == 16);

  // An unsaturated dump is the whole stream and may resume into a larger
  // capacity.
  SampleBuffer<TestRecord> small(16, 9);
  for (int64_t i = 0; i < 10; ++i) small.push({i});
  writeSamples(small, path);
  SampleBuffer<TestRecord> grown(32, 10);
  loadSamples(grown, path);
  REQUIRE(grown.totalSeen() == 10);
  REQUIRE(grown.size() == 10);

  std::remove(path.c_str());
}

// Splitting a stream across a save/load boundary must not bias retention:
// ids pushed before the dump and after the resume are retained at the same
// rate. This is the property that makes the reservoir persistent across CFR
// iterations.
void testUniformRetentionAcrossResume() {
  constexpr int kTrials = 3000;
  constexpr size_t kCapacity = 32;
  constexpr int kStreamSize = 256;
  const std::string path = tempPath("resume.bin");

  std::vector<long long> retained(kStreamSize, 0);
  for (int trial = 0; trial < kTrials; ++trial) {
    uint64_t seed = static_cast<uint64_t>(trial) * 0x9e3779b97f4a7c15ULL + 3;
    SampleBuffer<TestRecord> first(kCapacity, seed);
    for (int64_t id = 0; id < kStreamSize / 2; ++id) first.push({id});
    writeSamples(first, path);

    SampleBuffer<TestRecord> second(kCapacity, seed ^ 0xd1b54a32d192ed03ULL);
    loadSamples(second, path);
    for (int64_t id = kStreamSize / 2; id < kStreamSize; ++id) second.push({id});

    REQUIRE(second.totalSeen() == kStreamSize);
    for (const TestRecord& r : second.samples) ++retained[static_cast<size_t>(r.id)];
  }
  std::remove(path.c_str());

  const double expected = static_cast<double>(kTrials) * kCapacity / kStreamSize;
  for (int id = 0; id < kStreamSize; ++id) {
    REQUIRE(retained[id] > expected - 100 && retained[id] < expected + 100);
  }
}

void expectLoadFailure(SampleBuffer<TestRecord>& buffer, const std::string& path) {
  bool threw = false;
  try {
    loadSamples(buffer, path);
  } catch (const std::exception&) {
    threw = true;
  }
  REQUIRE(threw);
}

void testLoadValidation() {
  const std::string path = tempPath("validation.bin");

  // A saturated reservoir must not resume into a different capacity: the
  // retained set is a fixed-size sample of a longer stream.
  SampleBuffer<TestRecord> saturated(16, 11);
  for (int64_t i = 0; i < 100; ++i) saturated.push({i});
  writeSamples(saturated, path);
  SampleBuffer<TestRecord> wrongCapacity(32, 12);
  expectLoadFailure(wrongCapacity, path);

  // Loading into a non-empty buffer is a logic error.
  SampleBuffer<TestRecord> nonEmpty(16, 13);
  nonEmpty.push({0});
  expectLoadFailure(nonEmpty, path);

  // Version-1 files (three-field header, no version/seen) must be rejected.
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    uint64_t v1Header[3] = {SAMPLE_FILE_MAGIC, sizeof(TestRecord), 0};
    out.write(reinterpret_cast<const char*>(v1Header), sizeof(v1Header));
  }
  SampleBuffer<TestRecord> fresh(16, 14);
  expectLoadFailure(fresh, path);

  // Record-size drift must be rejected.
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    uint64_t header[5] = {SAMPLE_FILE_MAGIC, SAMPLE_FILE_VERSION, sizeof(TestRecord) + 8, 0, 0};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
  }
  expectLoadFailure(fresh, path);

  std::remove(path.c_str());
}

}  // namespace
}  // namespace pkrbot::engine

int main() {
  using namespace pkrbot::engine;
  testStagingFlush();
  testSaveLoadRoundtrip();
  testLoadValidation();
  testUniformRetentionUnderConcurrency();
  testUniformRetentionAcrossResume();
  std::cout << "all tests passed\n";
  return 0;
}
