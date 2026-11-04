"""Defaults baseline screening: 10 tasks x 4 episodes, separate from smoke."""
from pathlib import Path
import csv,json,os,subprocess,time
root=Path(__file__).resolve().parents[2]
import sys
name=sys.argv[1]
model=sys.argv[2]
flags=sys.argv[3:]
out=Path(__file__).resolve().parent/'screen40-variants'/name
out.mkdir(parents=True,exist_ok=True)
if (out/'protocol.json').exists():
    raise SystemExit('Output exists; inspect progress before resuming. No automatic overwrite.')
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
env.update(MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',LIBERO_CONFIG_PATH=str(Path(__file__).resolve().parent/'libero-config'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
server_cmd=[str(root/'build-cuda/vla-server'),'--bind','tcp://127.0.0.1:15560',model]+flags
protocol=dict(classification='screening, not official evaluation',suite='libero_object',tasks=list(range(10)),episodes_per_task=4,expected_total=40,seed=42,n_action_steps=1,variant=name,model=model,flags=flags,noise='repo defaults random server noise; environment seed42 only',baseline_reference='screen40-defaults:38/40; known bit-exact test failures disclosed',server=server_cmd,env={k:env[k] for k in ['MUJOCO_GL','CUDA_VISIBLE_DEVICES','LIBERO_CONFIG_PATH','TOKENIZERS_PARALLELISM']},initial_states='repo client defaults: episode_index=0, stride=1; same seed and initial states for later configs',aborts='halt on any aborted/missing episode; never silently reduce denominator')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
results=[]
def progress(state):
    data=dict(state=state,completed_tasks=len(results),completed_episodes=sum(r['episodes'] for r in results),successes=sum(r['successes'] for r in results),expected_episodes=40,tasks=results)
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
            cmd=[str(root/'eval/sim/libero/libero_uv/.venv/bin/python'),str(root/'eval/client/run_sim_client_direct.py'),'--arch','smolvla','--vla-addr','tcp://127.0.0.1:15560','--task','libero_object','--task-id',str(task),'--recv-timeout-ms','-1','--n-episodes','4','--n-action-steps','1','--seed','42','--output-dir',str(out/'outputs')]
            (out/f'task-{task}-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            print('Starting task',task,'4 episodes',flush=True)
            progress(f'running task {task}')
            with (out/f'task-{task}.log').open('w') as log:
                completed=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            if completed.returncode:raise RuntimeError(f'task {task} client exit {completed.returncode}')
            path=out/'outputs/smolvla/libero_object'/f'task_{task}'/'episodes.csv'
            with path.open() as f:episodes=list(csv.DictReader(f))
            if len(episodes)!=4 or any(int(r['aborted']) for r in episodes):raise RuntimeError(f'task {task}: incomplete/aborted episodes; see {path}')
            successes=sum(int(r['success']) for r in episodes)
            results.append(dict(task_id=task,episodes=4,successes=successes,steps=[int(r['steps']) for r in episodes],csv=str(path.relative_to(root))))
            data=progress('running')
            print('Completed task',task,f'{successes}/4; total {data["successes"]}/{data["completed_episodes"]}',flush=True)
            with (root/'NOTES_SMOLVLA.md').open('a') as notes:
                notes.write(f'\nScreen40 {name}: task {task} hoàn tất {successes}/4; tổng tạm thời {data["successes"]}/{data["completed_episodes"]}. Log screen40-variants/{name}/task-{task}.log; chưa phải official SR.\n')
        data=progress('complete')
        data['screening_sr']=data['successes']/40
        (out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    except BaseException as error:
        progress('failed: '+str(error))
        raise
    finally:
        server.terminate()
        try:server.wait(timeout=15)
        except subprocess.TimeoutExpired:server.kill();server.wait()
