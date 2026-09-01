#!/bin/bash
echo "=== INT OVERFLOWS ==="
rg "(int|int32_t)\s+[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_>.\-]*ne\[[0-3]\]\s*\*" src/
rg "(int|int32_t)\s+[a-zA-Z0-9_]+\s*=\s*.*\*.*ne\[[0-3]\]" src/
echo "=== FOPEN WITHOUT FCLOSE ==="
rg -l "fopen" src/ | xargs -I{} bash -c 'grep -q fclose {} || echo {} missing fclose'
echo "=== GGML_NEW_CONTEXT WITHOUT GGML_FREE ==="
rg -l "ggml_init" src/ | xargs -I{} bash -c 'grep -q ggml_free {} || echo {} missing ggml_free'
echo "=== BACKEND BUFFER MEMORY ==="
rg -l "ggml_backend_alloc_buffer" src/ | xargs -I{} bash -c 'grep -q ggml_backend_buffer_free {} || echo {} missing buffer free'
echo "=== SINGLE THREADED LOOPS ==="
rg -i "for.*y.*height.*for.*x.*width" src/
rg -i "for.*i.*<.*w.*h" src/
echo "=== F32 ROUND TRIPS ==="
rg "ggml_cast.*F32" src/
rg "ggml_cpy.*F32" src/
echo "=== UNCHECKED TENSOR SHAPES ==="
