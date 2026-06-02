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

#include "bitnet_kernels.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void bitlinear_unsupported_shape(const char * which,
                                                    int M, int N, int K) {
    std::fprintf(stderr,
        "vla(bitvla): %s has no specialised kernel for (M=%d, N=%d, K=%d). "
        "Register the shape in bitnet_kernels.cu and rebuild.\n",
        which, M, N, K);
    std::abort();
}

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, float* s, float* ws, int M, int N, int K, cudaStream_t stream){
    if (M == 1 && N == 3840 && K == 2560){
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 2560){
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 13824 && K == 2560){
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16><<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 6912){
        ladder_int8xint2_kernel<1, 2560, 6912, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 4800 && K == 3200){
        ladder_int8xint2_kernel<1, 4800, 3200, 6, 8, 16><<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 3200){
        ladder_int8xint2_kernel<1, 3200, 3200, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 20480 && K == 3200){
        ladder_int8xint2_kernel<1, 20480, 3200, 2, 8, 16><<<dim3(1280, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 10240){
        ladder_int8xint2_kernel<1, 3200, 10240, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 5120 && K == 27648){
        ladder_int8xint2_kernel<1, 5120, 27648, 1, 8, 16><<<dim3(320, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 55296 && K == 5120){
        ladder_int8xint2_kernel<1, 55296, 5120, 1, 8, 16><<<dim3(3456, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else{
        bitlinear_unsupported_shape("bitlinear_int8xint2", M, N, K);
    }
}

extern "C" void bitlinear_int8xint2_m(
    int8_t* input0, int8_t* input1,
    __nv_bfloat16* output0,
    float* s, float* ws,
    int M, int N, int K, cudaStream_t stream)
{

    if      (N == 2560  && K == 2560) launch_ladder_int8xint2_m<2560,  2560, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 640   && K == 2560) launch_ladder_int8xint2_m<640,   2560, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 13824 && K == 2560) launch_ladder_int8xint2_m<13824, 2560, 2, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 2560  && K == 6912) launch_ladder_int8xint2_m<2560,  6912, 1, 128>(input0, input1, output0, s, ws, M, stream);

    else if (N == 1152  && K == 1152) launch_ladder_int8xint2_m<1152,  1152, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 4304  && K == 1152) launch_ladder_int8xint2_m<4304,  1152, 1, 128>(input0, input1, output0, s, ws, M, stream);
    else if (N == 1152  && K == 4352) launch_ladder_int8xint2_m<1152,  4352, 1, 128>(input0, input1, output0, s, ws, M, stream);

    else if (N == 3840  && K == 2560) launch_ladder_int8xint2_m<3840,  2560, 3, 128>(input0, input1, output0, s, ws, M, stream);
    else
        bitlinear_unsupported_shape("bitlinear_int8xint2_m", M, N, K);
}

extern "C" void bitvla_act_quant_cuda(
    const __nv_bfloat16* in, int8_t* out, float* scales,
    int M, int K, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    act_quant_kernel<BLOCK_THREADS><<<dim3(M, 1, 1), dim3(BLOCK_THREADS, 1, 1), 0, stream>>>(in, out, scales, K);
}
