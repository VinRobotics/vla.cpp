"""Official defaults baseline: 10 tasks x 20 episodes; fixed denominator 200."""
from pathlib import Path
import csv,json,os,subprocess,time
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent/'official200-defaults'
out.mkdir(exist_ok=True)
if (out/'protocol.json').exists():
    raise SystemExit('Output exists; inspect progress before resuming. No automatic overwrite.')
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
env.update(MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',LIBERO_CONFIG_PATH=str(out.parent/'libero-config'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
server_cmd=[str(root/'build-cuda/vla-server'),'--bind','tcp://127.0.0.1:15559','/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf']
protocol=dict(classification='official evaluation',suite='libero_object',tasks=list(range(10)),episodes_per_task=20,expected_total=200,seed=42,n_action_steps=1,baseline='repo defaults; known bit-exact test failures disclosed',server=server_cmd,env={k:env[k] for k in ['MUJOCO_GL','CUDA_VISIBLE_DEVICES','LIBERO_CONFIG_PATH','TOKENIZERS_PARALLELISM']},initial_states='repo client defaults: episode_index=0, stride=1; same seed and initial states for later configs',aborts='halt on any aborted/missing episode; never silently reduce denominator')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
protocol['noise']='repo default server random RNG; seed42 controls environment only'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
results=[]
def progress(state):
    data=dict(state=state,completed_tasks=len(results),completed_episodes=sum(r['episodes'] for r in results),successes=sum(r['successes'] for r in results),expected_episodes=200,tasks=results)
    (out/'progress.json').write_text(json.dumps(data,indent=2)+'\n')
    return data
progress('starting')
with (out/'server.log').open('w') as server_log:
    server=subprocess.Popen(server_cmd,cwd=root,env=env,stdout=server_log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+180
        while 'ready.' not in (out/'server.log').read_text():
            if server.poll() is not None:raise RuntimeError('Server exited before ready')
            if time.monotonic()>deadline:raise TimeoutError('Server readiness timeout')
            time.sleep(.5)
        if 'backend = CUDA' not in (out/'server.log').read_text():raise RuntimeError('No CUDA backend confirmation')
        for task in range(10):
            cmd=[str(root/'eval/sim/libero/libero_uv/.venv/bin/python'),str(root/'eval/client/run_sim_client_direct.py'),'--arch','smolvla','--vla-addr','tcp://127.0.0.1:15559','--task','libero_object','--task-id',str(task),'--n-episodes','20','--n-action-steps','1','--seed','42','--output-dir',str(out/'outputs')]
            (out/f'task-{task}-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print('Starting task',task,'20 episodes',flush=True)
            progress(f'running task {task}')
            with (out/f'task-{task}.log').open('w') as log:
                completed=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
            if completed.returncode:raise RuntimeError(f'task {task} client exit {completed.returncode}')
            path=out/'outputs/smolvla/libero_object'/f'task_{task}'/'episodes.csv'
            with path.open() as f:episodes=list(csv.DictReader(f))
            if len(episodes)!=20 or any(int(r['aborted']) for r in episodes):raise RuntimeError(f'task {task}: incomplete/aborted episodes; see {path}')
            successes=sum(int(r['success']) for r in episodes)
            results.append(dict(task_id=task,episodes=20,successes=successes,steps=[int(r['steps']) for r in episodes],csv=str(path.relative_to(root))))
            data=progress('running')
            print('Completed task',task,f'{successes}/20; total {data["successes"]}/{data["completed_episodes"]}',flush=True)
            with (root/'NOTES_SMOLVLA.md').open('a') as notes:
                notes.write(f'\nOfficial200 defaults: task {task} hoàn tất {successes}/20; tổng tạm thời {data["successes"]}/{data["completed_episodes"]}. Log official200-defaults/task-{task}.log; SR tạm thời; chỉ chốt sau đủ200 episode.\n')
        data=progress('complete')
        data['official_sr']=data['successes']/200
        (out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    except BaseException as error:
        progress('failed: '+str(error))
        raise
    finally:
        server.terminate()
        try:server.wait(timeout=15)
        except subprocess.TimeoutExpired:server.kill();server.wait()
