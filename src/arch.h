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
 * @file arch.h
 * @brief Per-architecture model interface and factory declarations.
 *
 * Every supported VLA architecture implements @ref vla::ModelArchBase and is
 * created through a matching @c *_create factory in this header. Adding a new
 * architecture means: extend the @ref vla::Arch enum, declare a new factory
 * here, implement it under @c src/models/, and wire detection/dispatch in
 * @c src/model.cpp.
 */

#pragma once

#include "model.h"

#include <memory>
#include <string>

namespace vla {

/**
 * @brief Identifier for every VLA architecture the engine can serve.
 *
 * The value is detected from the GGUF checkpoint at load time
 * (@ref detect_arch_from_ckpt) and routed to the corresponding factory.
 */
enum class Arch {
    SMOLVLA,    // Hugging Face SmolVLA (mmproj + LM + flow-matching head).
    PI0,        // Physical Intelligence pi0 (PaliGemma + flow-matching).
    PI05,       // PaliGemma-3B + SigLIP-So400m + Gemma-300m + FM.
    EVO1,       // MINT-SJTU Evo-1 (InternVL3 + cross-attention head).
    GR00T_N1_5, // NVIDIA Isaac GR00T N1.5 (Eagle VLM + DiT action head).
    GR00T_N1_6, // NVIDIA Isaac GR00T N1.6 (Eagle Block-2A + DiT).
    GR00T_N1_7, // NVIDIA Isaac GR00T N1.7 (Qwen3 backbone + DiT).
    BITVLA,     // Microsoft BitVLA (1.58-bit ternary LM/ViT).
    VLA_ADAPTER,// OpenHelix VLA-Adapter DINOv2 + SigLIP + Bridge-Attention.
    OPENVLA_OFT,// DINOv2-L/14-reg4 + SigLIP-so400m/14 +Llama-2-7B + MLPResNet.
    VLA_JEPA,   // LeRobot Qwen3-VL-2B-Instruct+V-JEPÀ+DiT-B FM.
};

/**
 * @brief Common base for every concrete architecture implementation.
 *
 * Each subclass owns its llama.cpp contexts, vision tower, and any custom
 * CUDA state. Construction is performed through the per-arch
 * @c *_create factories declared below; the engine never instantiates
 * @ref ModelArchBase directly.
 */
class ModelArchBase {
public:
    Arch   arch;       ///< The architecture this instance implements.
    Config cfg{};      ///< Resolved model hyper-parameters (see @ref Config).
    Stats  stats{};    ///< Phase timings of the most recent @ref predict call.

    /**
     * @brief Construct a base for the given architecture.
     * @param a Arch tag the subclass implements.
     */
    explicit ModelArchBase(Arch a) : arch(a) {}
    virtual ~ModelArchBase() = default;

    /**
     * @brief Run a full forward pass and return one chunk of normalised actions.
     * @param in Vision + language + state inputs (see @ref Inputs).
     * @return Flattened action chunk of length
     *         @c cfg.num_steps * cfg.real_action_dim.
     */
    virtual std::vector<float> predict(const Inputs& in) = 0;
};

/**
 * @brief Build a SmolVLA model from its mmproj and checkpoint GGUFs.
 * @param mmproj_path Path to the vision-tower GGUF.
 * @param ckpt_path   Path to the LM + action-expert GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> smolvla_create(const std::string& mmproj_path,
                                              const std::string& ckpt_path,
                                              const std::string& config_path);

/**
 * @brief Build a pi0 model from its mmproj and checkpoint GGUFs.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> pi0_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string& config_path);

/**
 * @brief Build a pi0.5 model from its mmproj and checkpoint GGUFs.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> pi05_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& config_path);

/**
 * @brief Build an Evo-1 model. Vision is baked into @p ckpt_path; pass
 *        an empty @p mmproj_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> evo1_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& config_path);

/**
 * @brief Build a GR00T N1.5 model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> gr00t_n1_5_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& config_path);

/**
 * @brief Build a GR00T N1.6 model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> gr00t_n1_6_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& config_path);

/**
 * @brief Build a GR00T N1.7 model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> gr00t_n1_7_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& config_path);

/**
 * @brief Build a BitVLA model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> bitvla_create(const std::string& mmproj_path,
                                             const std::string& ckpt_path,
                                             const std::string& config_path);

/**
 * @brief Build a VLA-Adapter model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> vla_adapter_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string& config_path);

/**
 * @brief Build a OpenVLA-OFT model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> openvla_oft_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string& config_path);

/**
 * @brief Build a VLA-JEPA model. Vision is baked into @p ckpt_path.
 * @copydetails smolvla_create
 */
std::unique_ptr<ModelArchBase> vla_jepa_create(const std::string& mmproj_path,
                                                  const std::string& ckpt_path,
                                                  const std::string& config_path);

/**
 * @brief Inspect a GGUF and identify the architecture tag.
 *
 * Reads the GGUF metadata (without loading weights) and matches it against
 * the per-arch fingerprints declared in @c src/model.cpp.
 *
 * @param ckpt_path Path to the candidate GGUF.
 * @param[out] out  Receives the detected @ref Arch on success.
 * @return @c true if the architecture was recognised; @c false otherwise
 *         (and @p out is left untouched).
 */
bool detect_arch_from_ckpt(const std::string& ckpt_path, Arch* out);

}
