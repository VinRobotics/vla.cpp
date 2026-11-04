from pathlib import Path
import subprocess,os,json,time
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent/'smoke-original'
out.mkdir(exist_ok=True)
env=dict(os.environ,MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',LIBERO_CONFIG_PATH=str(out.parent/'libero-config'),PYTHONUNBUFFERED='1')
server_cmd=[str(root/'build-cuda/vla-server'),'--bind','tcp://127.0.0.1:15556','/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf']
client_cmd=[str(root/'eval/sim/libero/libero_uv/.venv/bin/python'),str(root/'eval/client/run_sim_client_direct.py'),'--arch','smolvla','--vla-addr','tcp://127.0.0.1:15556','--task','libero_object','--task-id','0','--n-episodes','2','--n-action-steps','1','--seed','42','--output-dir',str(out/'outputs')]
(out/'commands.json').write_text(json.dumps({'server':server_cmd,'client':client_cmd,'env':{k:env[k] for k in ['MUJOCO_GL','CUDA_VISIBLE_DEVICES','LIBERO_CONFIG_PATH']},'classification':'smoke only; not screening or official SR'},indent=2)+'\n')
with (out/'server.log').open('w') as slog:
 server=subprocess.Popen(server_cmd,cwd=root,env=env,stdout=slog,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+180
  while 'ready.' not in (out/'server.log').read_text():
   if server.poll() is not None:raise RuntimeError('Server exited before ready')
   if time.monotonic()>deadline:raise TimeoutError('Server readiness timeout')
   time.sleep(.5)
  print('Server CUDA ready; starting 2 smoke episodes',flush=True)
  with (out/'client.log').open('w') as clog:
   r=subprocess.run(client_cmd,cwd=root,env=env,stdout=clog,stderr=subprocess.STDOUT,timeout=1800)
  print('Client exit:',r.returncode,flush=True)
  (out/'exit.json').write_text(json.dumps({'client_exit':r.returncode})+'\n')
  if r.returncode:raise SystemExit(r.returncode)
 finally:
  server.terminate()
  try:server.wait(timeout=15)
  except subprocess.TimeoutExpired:server.kill();server.wait()
