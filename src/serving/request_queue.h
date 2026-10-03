// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The hand-off between vla-server's socket thread and its predict thread.
//
// The socket thread parses and validates a request and pushes a Job; the
// predict thread pops one at a time. A policy server only ever wants the newest
// observation from each robot, so in LATEST mode a push from a client that
// already has a job waiting replaces that job and hands it back to the caller,
// who answers it with a "superseded" error. FIFO keeps every request, bounded
// by a depth cap, for benchmarks that want throughput rather than freshness.
//
// Header-only and free of ZeroMQ, protobuf and the model so tests can drive it.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace vla::serving {

enum class QueueMode {
    LATEST,  ///< One pending job per client; a newer one displaces it.
    FIFO,    ///< Every job is served in arrival order, up to max_depth.
};

inline bool parse_queue_mode(const std::string & s, QueueMode & out) {
    if (s == "latest") { out = QueueMode::LATEST; return true; }
    if (s == "fifo")   { out = QueueMode::FIFO;   return true; }
    return false;
}

inline const char * queue_mode_name(QueueMode m) {
    return m == QueueMode::LATEST ? "latest" : "fifo";
}

/// What push() did with a job.
enum class PushResult {
    QUEUED,      ///< Appended; nothing displaced.
    REPLACED,    ///< Appended, and the same client's older pending job is in `displaced`.
    REJECTED,    ///< Queue full (FIFO only); the job itself is handed back in `displaced`.
};

template <class Job>
class RequestQueue {
public:
    explicit RequestQueue(QueueMode mode, size_t max_depth = 64)
        : mode_(mode), max_depth_(max_depth == 0 ? 1 : max_depth) {}

    QueueMode mode() const { return mode_; }

    /// Jobs are keyed by `key(job)`, the ZeroMQ routing envelope in the server.
    template <class KeyFn>
    PushResult push(Job job, KeyFn key, std::optional<Job> & displaced) {
        std::lock_guard<std::mutex> lk(mu_);
        displaced.reset();
        if (mode_ == QueueMode::LATEST) {
            const auto k = key(job);
            for (auto it = q_.begin(); it != q_.end(); ++it) {
                if (key(*it) == k) {
                    displaced = std::move(*it);
                    q_.erase(it);
                    q_.push_back(std::move(job));
                    cv_.notify_one();
                    return PushResult::REPLACED;
                }
            }
            // No pending job from this client. A bounded queue still protects
            // against a flood of distinct identities.
            if (q_.size() >= max_depth_) {
                displaced = std::move(job);
                return PushResult::REJECTED;
            }
            q_.push_back(std::move(job));
            cv_.notify_one();
            return PushResult::QUEUED;
        }
        if (q_.size() >= max_depth_) {
            displaced = std::move(job);
            return PushResult::REJECTED;
        }
        q_.push_back(std::move(job));
        cv_.notify_one();
        return PushResult::QUEUED;
    }

    /// Blocks until a job is available or stop() was called. Returns false on stop
    /// with the queue drained.
    bool pop(Job & out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stopped_ || !q_.empty(); });
        if (q_.empty())
            return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    /// Wakes pop(). Jobs still queued are returned by subsequent pops until
    /// empty, so the worker can answer them before it exits.
    void stop() {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    QueueMode               mode_;
    size_t                  max_depth_;
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<Job>         q_;
    bool                    stopped_ = false;
};

}  // namespace vla::serving
