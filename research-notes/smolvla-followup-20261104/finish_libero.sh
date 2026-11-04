#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/research-notes/tools/uv/bin:$PATH"
export UV_CACHE_DIR="$HOME/.cache/uv-vla"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=4
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export VIRTUAL_ENV="$PWD/eval/sim/libero/libero_uv/.venv"
uv pip install -e eval/sim/libero/LIBERO --config-settings editable_mode=compat
uv pip install research-notes/tools/cpu-wheels/*.whl
uv pip install pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install lerobot==0.4.3 --no-deps
uv pip install --requirements research-notes/smolvla-followup-20261104/lerobot-base-deps.txt
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install pandas==2.0.3
uv pip install pyarrow==12.0.1
uv pip install diffusers==0.30.1
uv pip install numpy==1.26.4
uv pip install mujoco==2.3.2
export LIBERO_CONFIG_PATH="$PWD/research-notes/smolvla-followup-20261104/libero-config"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
python3 - <<'PY'
from pathlib import Path
import json
root=Path('eval/sim/libero/LIBERO/libero/libero').resolve()
p=Path('research-notes/smolvla-followup-20261104/libero-config')
p.mkdir(exist_ok=True)
config={'benchmark_root':str(root),'bddl_files':str(root/'bddl_files'),'init_states':str(root/'init_files'),'datasets':str(root.parent/'datasets'),'assets':str(root/'assets')}
f=p/'config.yaml'
if not f.exists(): f.write_text(json.dumps(config,indent=2)+'\n')
PY
uv pip freeze > research-notes/smolvla-followup-20261104/libero-freeze.txt
uv pip check > research-notes/smolvla-followup-20261104/libero-pip-check.log 2>&1 || true
"$VIRTUAL_ENV/bin/python" research-notes/smolvla-followup-20261104/verify_libero.py > research-notes/smolvla-followup-20261104/libero-verify.log 2>&1
