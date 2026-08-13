from typing import Any
from utils.sim_adapters.base import BasePipelineAdapter


#################################################################################################
############################# PARSERS FOR DIFFERENT ARCHITECTURES ###############################
#################################################################################################
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


###############################################################################################
########################## ADAPTER THAT USES THE PARSERS TO INTERFACE WITH SIM ################
###############################################################################################
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
