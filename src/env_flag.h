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

/**
 * @file env_flag.h
 * @brief Boolean environment switches with a value, not just presence.
 */

#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace vla {

/**
 * @brief Read a boolean environment switch.
 *
 * Presence alone used to mean "on", so @c VLA_EVO1_FA=0 turned flash attention
 * *on* -- the opposite of what anyone typing it intends, and a quiet way to
 * publish a benchmark of the wrong configuration.
 *
 * @c 0, @c false, @c off, @c no and the empty string are false (case
 * insensitive); any other value is true; unset returns @p def. This matches
 * @c VLA_GR00T_GRAPH_CACHE, which already parsed its value rather than its
 * presence.
 *
 * @param name Variable to read.
 * @param def  Value to use when the variable is not set at all.
 */
inline bool env_flag(const char * name, bool def = false) {
    const char * v = std::getenv(name);
    if (!v)
        return def;
    if (!*v) return false;          // FOO= reads as "unset it"

    char   buf[8] = {};
    size_t n      = 0;
    for (; n < sizeof(buf) - 1 && v[n]; ++n) {
        buf[n] = (char) std::tolower((unsigned char) v[n]);
    }
    if (v[n]) return true;          // longer than any false word, so it is one

    return !(std::strcmp(buf, "0")     == 0 ||
             std::strcmp(buf, "false") == 0 ||
             std::strcmp(buf, "off")   == 0 ||
             std::strcmp(buf, "no")    == 0);
}

}  // namespace vla
