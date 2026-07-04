// LibTorch implementation of the InferenceEngine seam (net_policy.h).
//
// Loads the TorchScript module exported by export_torchscript()
// (train/model.py) and runs it on CUDA when available, else CPU. The export
// embeds feature_version / input_size / output_size as extra files; they are
// checked against this build's encoding.h so a stale export fails loudly at
// startup instead of silently mis-slicing the input tensor.
//
// Threading: infer() is only ever called from the batcher's single inference
// thread (inference.h), so the module and the default CUDA stream need no
// locking.

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <torch/cuda.h>
#include <torch/script.h>

#include "encoding.h"
#include "net_policy.h"

namespace pkrbot::engine {

namespace {

class TorchEngine : public InferenceEngine {
 public:
  TorchEngine(torch::jit::Module module, torch::Device device, int maxBatchSize)
      : module_(std::move(module)), device_(device), maxBatchSize_(maxBatchSize) {}

  int inputSize() const override { return INPUT_SIZE; }
  int outputSize() const override { return NUM_ABSTRACT_ACTIONS; }
  int maxBatchSize() const override { return maxBatchSize_; }

  void infer(const float* input, float* output, int batch) override {
    torch::InferenceMode guard;
    torch::Tensor in =
        torch::from_blob(const_cast<float*>(input), {batch, INPUT_SIZE}, torch::kFloat32)
            .to(device_);
    torch::Tensor out = module_.forward({in}).toTensor().to(torch::kCPU).contiguous();
    std::memcpy(output, out.const_data_ptr<float>(),
                static_cast<size_t>(batch) * NUM_ABSTRACT_ACTIONS * sizeof(float));
  }

 private:
  torch::jit::Module module_;
  torch::Device device_;
  int maxBatchSize_;
};

int metadataInt(const std::unordered_map<std::string, std::string>& extra, const std::string& key,
                const std::string& enginePath) {
  auto it = extra.find(key);
  if (it == extra.end() || it->second.empty()) {
    throw std::runtime_error(enginePath + ": missing '" + key +
                             "' metadata; re-export with export_torchscript() (train/model.py)");
  }
  return std::stoi(it->second);
}

}  // namespace

std::unique_ptr<InferenceEngine> loadInferenceEngine(const std::string& enginePath,
                                                     int maxBatchSize) {
  std::unordered_map<std::string, std::string> extra = {
      {"feature_version", ""}, {"input_size", ""}, {"output_size", ""}};
  torch::jit::Module module = torch::jit::load(enginePath, torch::kCPU, extra);

  int featureVersion = metadataInt(extra, "feature_version", enginePath);
  if (featureVersion != FEATURE_VERSION) {
    throw std::runtime_error(enginePath + ": feature version " + std::to_string(featureVersion) +
                             " != this build's " + std::to_string(FEATURE_VERSION));
  }
  if (metadataInt(extra, "input_size", enginePath) != INPUT_SIZE ||
      metadataInt(extra, "output_size", enginePath) != NUM_ABSTRACT_ACTIONS) {
    throw std::runtime_error(enginePath + ": input/output size does not match encoding.h");
  }

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  module.eval();
  module.to(device);
  return std::make_unique<TorchEngine>(std::move(module), device, maxBatchSize);
}

}  // namespace pkrbot::engine
