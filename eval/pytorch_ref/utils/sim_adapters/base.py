from typing import Any

class BasePipelineAdapter:
    def __init__(self, client: Any, parser: Any, arch: str):
        self._client = client
        self._parser = parser
        self.arch = arch

    def reset(self):
        return self._client.reset()
    
    def get_action(self, obs: dict[str, Any]) -> Any:
        parsed_obs = self._parser.parse_observation(obs)
        action = self._client.get_action(parsed_obs)
        parsed_action = self._parser.parse_action(action)
        return parsed_action
