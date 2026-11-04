"""Regenerates tests/data/rtc_reference.inc from LeRobot's RTCProcessor (needs lerobot + torch):
    python tests/gen_rtc_reference.py > ref.json   # then convert to the .inc layout used by test_prefix_cond.cpp
The .inc holds get_prefix_weights for every schedule and the guidance weight clamp at several t."""
import json, torch, math
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.configs.types import RTCAttentionSchedule as S
out = {"weights": [], "gw": []}
for name in ("ZEROS", "ONES", "LINEAR", "EXP"):
    p = RTCProcessor(RTCConfig(prefix_attention_schedule=getattr(S, name), execution_horizon=10, max_guidance_weight=10.0))
    for (start, end, total) in ((0, 10, 50), (3, 10, 50), (3, 7, 10), (5, 5, 10), (0, 4, 10), (7, 10, 10), (2, 12, 10)):
        out["weights"].append({"schedule": name, "start": start, "end": end, "total": total,
                               "w": [float(x) for x in p.get_prefix_weights(start, end, total)]})
for mgw in (10.0, 3.0):
    for t in (1.0, 0.9, 0.5, 0.3, 0.1, 0.05):
        tau = 1 - t
        tau_t = torch.as_tensor(tau); mx = torch.as_tensor(mgw)
        sq = (1 - tau_t) ** 2; inv_r2 = (sq + tau_t ** 2) / sq
        c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=mx)
        gw = torch.minimum(torch.nan_to_num(c * inv_r2, posinf=mx), mx)
        out["gw"].append({"t": t, "max": mgw, "gw": float(gw)})
print(json.dumps(out))
