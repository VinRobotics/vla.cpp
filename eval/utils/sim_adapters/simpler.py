# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from utils.sim_adapters.base import BasePipelineAdapter

class GR00TN16SimplerParser:

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        new_obs = {"video": {}, "state": {}, "language": {}}

        for key, value in obs.items():
            if key == "annotation.human.action.task_description":
                new_obs["language"][key] = [[value]]
            elif key.startswith("video."):
                new_obs["video"][key[len("video."):]] = value[None, None]
            elif key.startswith("state."):
                new_obs["state"][key[len("state."):]] = value[None, None]

        return new_obs

    def parse_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return {f"action.{key}": value[0][0] for key, value in action.items()}

SIMPLER_PARSER_REGISTRY = {
    "gr00t": GR00TN16SimplerParser,
}

class SimplerSimAdapter(BasePipelineAdapter):
    def __init__(self, client: Any):
        arch = client.get_arch()
        parser_cls = SIMPLER_PARSER_REGISTRY.get(arch)
        if parser_cls is None:
            raise ValueError(f"No parser found for architecture {arch}")
        super().__init__(client=client, parser=parser_cls(), arch=arch)
