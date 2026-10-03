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

// The rules vla-server relies on: in LATEST mode a client's newer request
// displaces its pending one and nobody else's; FIFO keeps order and caps depth;
// stop() lets the worker drain what is queued and then returns false.

#include "serving/request_queue.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using vla::serving::PushResult;
using vla::serving::QueueMode;
using vla::serving::RequestQueue;

struct Job {
    std::string client;
    int         rid = 0;
};

static const std::string & key_of(const Job & j) { return j.client; }

static void test_latest_replaces_same_client_only() {
    RequestQueue<Job> q(QueueMode::LATEST);
    std::optional<Job> displaced;

    assert(q.push(Job{"a", 1}, key_of, displaced) == PushResult::QUEUED);
    assert(!displaced);
    assert(q.push(Job{"b", 2}, key_of, displaced) == PushResult::QUEUED);
    assert(!displaced);
    assert(q.size() == 2);

    // a's second request displaces a's first, and is served after b's.
    assert(q.push(Job{"a", 3}, key_of, displaced) == PushResult::REPLACED);
    assert(displaced && displaced->client == "a" && displaced->rid == 1);
    assert(q.size() == 2);

    Job j;
    assert(q.pop(j) && j.client == "b" && j.rid == 2);
    assert(q.pop(j) && j.client == "a" && j.rid == 3);
    assert(q.size() == 0);
}

static void test_latest_does_not_touch_a_popped_job() {
    // A job already handed to the worker is being predicted; a newer request
    // from the same client queues behind it rather than cancelling it.
    RequestQueue<Job> q(QueueMode::LATEST);
    std::optional<Job> displaced;
    Job j;

    q.push(Job{"a", 1}, key_of, displaced);
    assert(q.pop(j) && j.rid == 1);
    assert(q.push(Job{"a", 2}, key_of, displaced) == PushResult::QUEUED);
    assert(!displaced);
    assert(q.pop(j) && j.rid == 2);
}

static void test_fifo_keeps_order_and_caps_depth() {
    RequestQueue<Job> q(QueueMode::FIFO, /*max_depth=*/2);
    std::optional<Job> displaced;

    assert(q.push(Job{"a", 1}, key_of, displaced) == PushResult::QUEUED);
    assert(q.push(Job{"a", 2}, key_of, displaced) == PushResult::QUEUED);
    assert(q.push(Job{"a", 3}, key_of, displaced) == PushResult::REJECTED);
    // The rejected job is the new one, handed back so the caller can answer it.
    assert(displaced && displaced->rid == 3);
    assert(q.size() == 2);

    Job j;
    assert(q.pop(j) && j.rid == 1);
    assert(q.pop(j) && j.rid == 2);
}

static void test_latest_caps_distinct_clients() {
    RequestQueue<Job> q(QueueMode::LATEST, /*max_depth=*/2);
    std::optional<Job> displaced;

    assert(q.push(Job{"a", 1}, key_of, displaced) == PushResult::QUEUED);
    assert(q.push(Job{"b", 2}, key_of, displaced) == PushResult::QUEUED);
    assert(q.push(Job{"c", 3}, key_of, displaced) == PushResult::REJECTED);
    assert(displaced && displaced->client == "c");
    // A known client still gets to replace its own job when the queue is full.
    assert(q.push(Job{"a", 4}, key_of, displaced) == PushResult::REPLACED);
    assert(displaced && displaced->rid == 1);
}

static void test_stop_drains_then_returns_false() {
    RequestQueue<Job> q(QueueMode::LATEST);
    std::optional<Job> displaced;
    q.push(Job{"a", 1}, key_of, displaced);
    q.push(Job{"b", 2}, key_of, displaced);
    q.stop();

    Job j;
    assert(q.pop(j) && j.rid == 1);
    assert(q.pop(j) && j.rid == 2);
    assert(!q.pop(j));
    assert(!q.pop(j));
}

static void test_pop_blocks_until_push_or_stop() {
    RequestQueue<Job> q(QueueMode::LATEST);
    std::vector<int> seen;

    std::thread worker([&] {
        Job j;
        while (q.pop(j))
            seen.push_back(j.rid);
    });

    std::optional<Job> displaced;
    for (int i=1; i<=5; ++i)
        q.push(Job{"a" + std::to_string(i), i}, key_of, displaced);
    q.stop();
    worker.join();

    assert(seen.size() == 5);
    for (int i=0; i<5; ++i)
        assert(seen[i] == i+1);
}

int main() {
    test_latest_replaces_same_client_only();
    test_latest_does_not_touch_a_popped_job();
    test_fifo_keeps_order_and_caps_depth();
    test_latest_caps_distinct_clients();
    test_stop_drains_then_returns_false();
    test_pop_blocks_until_push_or_stop();
    std::printf("test_request_queue: OK\n");
    return 0;
}
