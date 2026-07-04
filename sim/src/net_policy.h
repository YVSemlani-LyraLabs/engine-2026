#pragma once

// Connective layer between the batcher and the neural-net inference engine.
//
// NetPolicy does everything on the C++ side that is independent of the
// inference runtime: it owns host staging buffers sized to the batch limit,
// encodes InfoSet batches into the float32 [batch, INPUT_SIZE] input tensor
// (encoding.h), and decodes the [batch, NUM_ABSTRACT_ACTIONS] advantage
// output back into policies by regret matching.
//
// InferenceEngine is the runtime seam. The current implementation
// (torch_engine.cpp) wraps a TorchScript module exported by
// train/train_advantage.py, loaded via libtorch and run on CUDA when
// available. It is only ever called from the batcher's single inference
// thread (see inference.h), so one module + stream with no internal locking
// is sufficient. Swapping in a faster runtime later (e.g. TensorRT) only
// means reimplementing loadInferenceEngine().

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "abstraction.h"
#include "encoding.h"
#include "infoset.h"
#include "policy.h"

namespace pkrbot::engine {

// Minimal surface the runtime wrapper must implement. Kept free of runtime
// types so this header compiles everywhere; the implementation lives in
// torch_engine.cpp, compiled only into targets that link libtorch.
struct InferenceEngine {
  virtual ~InferenceEngine() = default;

  virtual int inputSize() const = 0;
  virtual int outputSize() const = 0;
  virtual int maxBatchSize() const = 0;

  // Run the network on `batch` rows: input is [batch, inputSize()] floats,
  // output is [batch, outputSize()] floats, both row-major on the host.
  // Called only from the batcher's inference thread.
  virtual void infer(const float* input, float* output, int batch) = 0;
};

// Load the TorchScript module exported by export_torchscript() (train/model.py)
// and wrap it as an InferenceEngine. Rejects modules whose embedded
// feature-version/shape metadata does not match this build's encoding.h.
// `maxBatchSize` bounds the per-call batch and sizes NetPolicy's staging
// buffers; pass BatchingConfig::maxBatchSize.
std::unique_ptr<InferenceEngine> loadInferenceEngine(const std::string& enginePath,
                                                     int maxBatchSize);

// PolicyProvider that evaluates the advantage net. Hand it to the
// InferenceBatcher as the backend and the traversal workers never know the
// difference from UniformPolicy.
class NetPolicy : public PolicyProvider {
 public:
  explicit NetPolicy(std::unique_ptr<InferenceEngine> engine) : engine_(std::move(engine)) {
    if (engine_->inputSize() != INPUT_SIZE) {
      throw std::runtime_error("engine input size " + std::to_string(engine_->inputSize()) +
                               " != encoder INPUT_SIZE " + std::to_string(INPUT_SIZE) +
                               " (feature version " + std::to_string(FEATURE_VERSION) + ")");
    }
    if (engine_->outputSize() != NUM_ABSTRACT_ACTIONS) {
      throw std::runtime_error("engine output size " + std::to_string(engine_->outputSize()) +
                               " != NUM_ABSTRACT_ACTIONS");
    }
    input_.resize(static_cast<size_t>(engine_->maxBatchSize()) * INPUT_SIZE);
    output_.resize(static_cast<size_t>(engine_->maxBatchSize()) * NUM_ABSTRACT_ACTIONS);
  }

  void evaluate(const InfoSet* obs, size_t n, PolicyVector* out) override {
    // The batcher caps batches at BatchingConfig::maxBatchSize, but that knob
    // is tuned independently of the engine's limit; split defensively so an
    // oversized request never reaches the engine.
    size_t chunk = static_cast<size_t>(engine_->maxBatchSize());
    for (size_t start = 0; start < n; start += chunk) {
      size_t count = std::min(chunk, n - start);
      for (size_t i = 0; i < count; ++i) {
        encodeInfoSet(obs[start + i], input_.data() + i * INPUT_SIZE);
      }
      engine_->infer(input_.data(), output_.data(), static_cast<int>(count));
      for (size_t i = 0; i < count; ++i) {
        out[start + i] = policyFromAdvantages(output_.data() + i * NUM_ABSTRACT_ACTIONS,
                                              obs[start + i].legalActionMask);
      }
    }
  }

 private:
  std::unique_ptr<InferenceEngine> engine_;
  // Host staging buffers, reused across calls. Safe without locking because
  // the batcher serializes backend calls onto its inference thread.
  std::vector<float> input_;
  std::vector<float> output_;
};

}  // namespace pkrbot::engine
