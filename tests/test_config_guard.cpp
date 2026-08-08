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

// predict() sizes host buffers from the max_* dims and loops to the real_* dims,
// so real > max writes out of bounds. SmolVLA shipped that loop unbounded.

#include "model.h"

#undef NDEBUG  // keep assert() live even in Release builds
#include <cassert>
#include <cstdio>

// Mirrors config_is_sane() in src/model.cpp.
static bool sane(const vla::Config & c) {
    if (c.real_state_dim  < 0 || c.max_state_dim  < 0) return false;
    if (c.real_action_dim < 0 || c.max_action_dim < 0) return false;
    if (c.max_state_dim  > 0 && c.real_state_dim  > c.max_state_dim)  return false;
    if (c.max_action_dim > 0 && c.real_action_dim > c.max_action_dim) return false;
    return true;
}

int main() {
    vla::Config c{};

    // A realistic SmolVLA config passes.
    c.max_state_dim = 32; c.real_state_dim = 8;
    c.max_action_dim = 32; c.real_action_dim = 7;
    assert(sane(c));

    // Equal is fine: the loops are half-open.
    c.real_state_dim = 32; c.real_action_dim = 32;
    assert(sane(c));

    // real > max is the heap-overflow shape.
    c.real_state_dim = 33;
    assert(!sane(c));
    c.real_state_dim = 8;
    c.real_action_dim = 33;
    assert(!sane(c));
    c.real_action_dim = 7;
    assert(sane(c));

    // Negative dims come from a garbage or hostile GGUF.
    c.real_state_dim = -1;
    assert(!sane(c));
    c.real_state_dim = 8;
    c.max_action_dim = -1;
    assert(!sane(c));
    c.max_action_dim = 32;
    assert(sane(c));

    // max == 0 means "arch does not use this dim"; do not reject those.
    c.max_state_dim = 0; c.real_state_dim = 0;
    assert(sane(c));

    std::printf("config guard: ok\n");
    return 0;
}
