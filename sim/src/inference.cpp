#include "inference.h"

namespace pkrbot::engine {

InferenceBatcher::InferenceBatcher(PolicyProvider& backend, const BatchingConfig& config,
                                   int numWorkers)
    : backend_(backend), config_(config), liveWorkers_(numWorkers) {
  inferenceThread_ = std::thread([this] { inferenceLoop(); });
}

InferenceBatcher::~InferenceBatcher() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    shutdown_ = true;
  }
  requestCv_.notify_all();
  inferenceThread_.join();
}

PolicyVector InferenceBatcher::evaluate(const InfoSet& obs) {
  Request request;
  request.obs = &obs;

  std::unique_lock<std::mutex> lock(mu_);
  pending_.push_back(&request);
  requestCv_.notify_one();
  resultCv_.wait(lock, [&request] { return request.ready; });
  return request.result;
}

void InferenceBatcher::workerFinished() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    --liveWorkers_;
  }
  // One fewer worker can submit, so "everyone still running is blocked" may
  // hold now; wake the inference thread to re-check.
  requestCv_.notify_one();
}

BatcherStats InferenceBatcher::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

void InferenceBatcher::inferenceLoop() {
  // Reused across batches to avoid reallocating per dispatch.
  std::vector<Request*> batch;
  std::vector<InfoSet> batchObs;
  std::vector<PolicyVector> batchOut;
  batch.reserve(config_.maxBatchSize);
  batchObs.reserve(config_.maxBatchSize);
  batchOut.reserve(config_.maxBatchSize);

  std::unique_lock<std::mutex> lock(mu_);
  while (true) {
    requestCv_.wait(lock, [this] { return !pending_.empty() || shutdown_; });
    if (pending_.empty()) {
      if (shutdown_) return;
      continue;
    }

    // A batch starts forming at the first pending request; give it up to
    // flushTimeout to fill unless a flush condition fires earlier.
    auto deadline = std::chrono::steady_clock::now() + config_.flushTimeout;
    auto flushable = [this] {
      return shutdown_ || static_cast<int>(pending_.size()) >= config_.maxBatchSize ||
             static_cast<int>(pending_.size()) >= liveWorkers_;
    };
    bool flushedEarly = requestCv_.wait_until(lock, deadline, flushable);

    if (static_cast<int>(pending_.size()) >= config_.maxBatchSize) {
      ++stats_.fullBatches;
    } else if (flushedEarly && !shutdown_) {
      ++stats_.starvedBatches;
    } else if (!flushedEarly) {
      ++stats_.timeoutBatches;
    }

    // Snapshot up to maxBatchSize requests. Infosets are copied into a
    // contiguous buffer because the backend takes an InfoSet array, while
    // requests point at per-worker stack storage.
    size_t n = std::min(pending_.size(), static_cast<size_t>(config_.maxBatchSize));
    batch.clear();
    batchObs.clear();
    for (size_t i = 0; i < n; ++i) {
      batch.push_back(pending_.front());
      batchObs.push_back(*pending_.front()->obs);
      pending_.pop_front();
    }
    stats_.requests += static_cast<long long>(n);
    ++stats_.batches;
    stats_.maxBatch = std::max(stats_.maxBatch, static_cast<int>(n));

    // Run the backend without holding the lock so workers can queue the next
    // batch while this one is in flight.
    lock.unlock();
    batchOut.resize(n);
    backend_.evaluate(batchObs.data(), n, batchOut.data());
    lock.lock();

    for (size_t i = 0; i < n; ++i) {
      batch[i]->result = batchOut[i];
      batch[i]->ready = true;
    }
    resultCv_.notify_all();
  }
}

}  // namespace pkrbot::engine
