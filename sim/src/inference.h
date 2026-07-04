#pragma once

// Batched inference for MCCFR traversals.
//
// A traversal is depth-first and needs exactly one policy evaluation at a
// time, so a single traversal can never fill a GPU batch on its own. Batching
// therefore comes from concurrency across traversals: the train driver runs
// many traversal workers, each worker blocks in InferenceBatcher::evaluate()
// at its decision points, and a dedicated inference thread gathers the blocked
// workers' infosets into one backend call (UniformPolicy, or the advantage
// net behind NetPolicy in net_policy.h).
//
// A batch is dispatched when the first of these fires:
//   1. maxBatchSize requests are pending (a full batch);
//   2. every live worker is blocked waiting (no further request can arrive,
//      so waiting longer only adds latency); or
//   3. flushTimeout has elapsed since the batch started forming (safety net
//      against stragglers that are busy in game logic).
//
// maxBatchSize and flushTimeout are hyperparameters (BatchingConfig), surfaced
// on the train CLI so they can be swept like any other hyperparameter and
// tuned against the real inference engine. The effective
// batch size is also capped by the worker count -- each blocked worker
// contributes exactly one pending request -- so tune workers alongside batch
// size. BatcherStats reports the realized batch-size distribution and which
// condition triggered each flush, which is the signal to tune against.
//
// Threading contract: evaluate() may be called from any number of worker
// threads; the backend's PolicyProvider::evaluate() is only ever invoked from
// the batcher's single inference thread (and never under the batcher's lock),
// so backends may hold non-thread-safe state such as a CUDA stream and
// execution context.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "infoset.h"
#include "policy.h"

namespace pkrbot::engine {

struct BatchingConfig {
  // Maximum infosets per backend call; also sizes the engine's staging
  // buffers (see net_policy.h).
  int maxBatchSize = 256;

  // How long a partially-filled batch may wait for more requests before it is
  // flushed anyway. Irrelevant for the CPU UniformPolicy backend; against a
  // GPU backend this trades per-call efficiency (bigger batches) for worker
  // latency.
  std::chrono::microseconds flushTimeout{250};
};

struct BatcherStats {
  long long requests = 0;
  long long batches = 0;
  long long fullBatches = 0;     // flushed at maxBatchSize
  long long starvedBatches = 0;  // flushed because all live workers were waiting
  long long timeoutBatches = 0;  // flushed by flushTimeout
  int maxBatch = 0;

  double avgBatch() const {
    return batches > 0 ? static_cast<double>(requests) / static_cast<double>(batches) : 0.0;
  }
};

class InferenceBatcher {
 public:
  // `backend` must outlive the batcher. `numWorkers` is the number of threads
  // that will call evaluate(); each must call workerFinished() exactly once
  // when it will submit no more requests, otherwise partial batches from the
  // remaining workers wait on the flush timeout instead of dispatching as
  // soon as everyone still running is blocked.
  InferenceBatcher(PolicyProvider& backend, const BatchingConfig& config, int numWorkers);
  ~InferenceBatcher();

  InferenceBatcher(const InferenceBatcher&) = delete;
  InferenceBatcher& operator=(const InferenceBatcher&) = delete;

  // Blocks the calling worker until its infoset has been evaluated as part of
  // a batch. `obs` only needs to stay alive for the duration of the call.
  PolicyVector evaluate(const InfoSet& obs);

  // Marks one worker as done submitting requests (see constructor).
  void workerFinished();

  BatcherStats stats() const;

 private:
  struct Request {
    const InfoSet* obs;
    PolicyVector result{};
    bool ready = false;
  };

  void inferenceLoop();

  PolicyProvider& backend_;
  const BatchingConfig config_;

  mutable std::mutex mu_;
  std::condition_variable requestCv_;  // wakes the inference thread
  std::condition_variable resultCv_;   // wakes workers whose results are ready
  std::deque<Request*> pending_;
  int liveWorkers_;
  bool shutdown_ = false;
  BatcherStats stats_;

  std::thread inferenceThread_;
};

// PolicyProvider adapter so the traversal drives the batcher without any
// change to traverse(): each worker owns one BatchedPolicy and every
// single-infoset evaluate() call becomes a blocking submission to the shared
// batcher, where it is coalesced with the other workers' requests.
struct BatchedPolicy : PolicyProvider {
  explicit BatchedPolicy(InferenceBatcher& batcher) : batcher_(batcher) {}

  void evaluate(const InfoSet* obs, size_t n, PolicyVector* out) override {
    for (size_t i = 0; i < n; ++i) {
      out[i] = batcher_.evaluate(obs[i]);
    }
  }

 private:
  InferenceBatcher& batcher_;
};

}  // namespace pkrbot::engine
