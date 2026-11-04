"""Run after installing LIBERO with MUJOCO_GL=egl and LIBERO_CONFIG_PATH set."""
import importlib.metadata as md
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

print('Python:',sys.version,flush=True)
for package in ['torch','torchvision','transformers','numpy','mujoco','robosuite','lerobot','gymnasium','pyarrow']:
 print(package,md.version(package),flush=True)
print('torch CUDA available:',torch.cuda.is_available(),flush=True)
from eval.client import adapters, vla_cpp_client
print('repo client/adapters imports OK',flush=True)
suite=benchmark.get_benchmark_dict()['libero_object']()
task=suite.get_task(0)
print('task count:',suite.n_tasks,'task:',task.name,flush=True)
bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
print('bddl:',bddl,'exists:',bddl.exists(),flush=True)
env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=128,camera_widths=128)
try:
 env.seed(42)
 obs=env.reset()
 frames={k:list(v.shape) for k,v in obs.items() if k.endswith('_image')}
 print('render frames:',json.dumps(frames),flush=True)
 obs,reward,done,info=env.step([0,0,0,0,0,0,-1])
 print('step OK, reward:',reward,'done:',done,flush=True)
 assert frames and all(np.isfinite(obs[k]).all() for k in frames)
finally:
 env.close()
print('LIBERO_INSTALL_AND_RENDER_OK',flush=True)
