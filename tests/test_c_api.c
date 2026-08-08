/* Copyright 2026 VinRobotics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Compiled as C, not C++, so it fails if the header ever stops being C-clean.
 * Argument checks run with no model. Set VLA_TEST_GGUF to also run a real
 * predict and compare against vla-cli on the same inputs. */

#include "vla.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int cond, const char * what) {
    if (!cond) { printf("FAIL: %s\n", what); failures++; }
}

int main(void) {
    check(vla_abi_version() == VLA_ABI_VERSION, "abi version matches header");

    /* Null handling must not crash. */
    vla_model_free(NULL);
    vla_free_actions(NULL);
    check(vla_model_load(NULL, NULL, NULL) == NULL, "load(NULL) returns NULL");

    vla_config cfg;
    check(vla_model_config(NULL, &cfg) == VLA_ERR_ARG, "config(NULL) is ERR_ARG");
    vla_stats st;
    check(vla_last_stats(NULL, &st) == VLA_ERR_ARG, "stats(NULL) is ERR_ARG");

    float * act = NULL;
    int64_t n = 0;
    vla_inputs in;
    memset(&in, 0, sizeof(in));
    check(vla_predict(NULL, &in, &act, &n) == VLA_ERR_ARG, "predict(NULL) is ERR_ARG");

    const char * gguf = getenv("VLA_TEST_GGUF");
    if (!gguf) {
        printf("c api: ok (argument checks only; set VLA_TEST_GGUF for a real run)\n");
        return failures ? 1 : 0;
    }

    vla_model * m = vla_model_load(getenv("VLA_TEST_MMPROJ"), gguf, NULL);
    check(m != NULL, "load real checkpoint");
    if (!m) return 1;

    check(vla_model_config(m, &cfg) == VLA_OK, "config on a loaded model");
    printf("c api: max_action_dim=%lld n_suffix=%lld denormalized=%d\n",
           (long long) cfg.max_action_dim, (long long) cfg.n_suffix, cfg.denormalized);

    /* Same synthetic inputs predict_check uses, so the numbers are comparable. */
    int side = getenv("VLA_IMG_SIZE") ? atoi(getenv("VLA_IMG_SIZE")) : 224;
    unsigned char * px = (unsigned char *) malloc((size_t) 3 * side * side);
    check(px != NULL, "pixel buffer");
    if (!px) return 1;
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
            for (int c = 0; c < 3; ++c)
                px[((size_t) y * side + x) * 3 + c] = (unsigned char) ((x + 2 * y + 40 * c) & 0xFF);

    vla_image img;
    img.data = px; img.w = side; img.h = side; img.format = VLA_PIXEL_U8;

    int32_t lang[6] = { 1, 100, 200, 300, 400, 2 };
    float * state = (float *) calloc((size_t) (cfg.max_state_dim > 0 ? cfg.max_state_dim : 1), sizeof(float));
    for (int i = 0; i < (int) cfg.real_state_dim; ++i) state[i] = 0.01f * (float) (i + 1);

    size_t noise_n = (size_t) cfg.max_action_dim * (size_t) cfg.n_suffix;
    float * noise = noise_n ? (float *) malloc(noise_n * sizeof(float)) : NULL;
    for (size_t i = 0; i < noise_n; ++i)
        noise[i] = 0.001f * (float) ((i * 2654435761u) % 1000) - 0.5f;

    memset(&in, 0, sizeof(in));
    in.images = &img;
    in.n_images = 1;
    in.lang_tokens = lang;
    in.n_lang = 6;
    in.state = state;
    in.noise = noise;

    int32_t rc = vla_predict(m, &in, &act, &n);
    check(rc == VLA_OK, "predict returns OK");
    check(act != NULL && n > 0, "predict returns a buffer");
    if (rc == VLA_OK) {
        printf("action_len=%lld\n", (long long) n);
        for (int64_t i = 0; i < n; ++i) printf("%.9g\n", (double) act[i]);
        vla_free_actions(act);
    }

    check(vla_last_stats(m, &st) == VLA_OK, "stats after predict");
    check(st.ms_total > 0.0f, "total time is positive");

    free(noise); free(state); free(px);
    vla_model_free(m);

    if (failures) { printf("c api: %d FAILURES\n", failures); return 1; }
    printf("c api: ok\n");
    return 0;
}
