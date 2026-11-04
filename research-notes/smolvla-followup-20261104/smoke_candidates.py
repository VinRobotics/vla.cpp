"""Two LIBERO smoke episodes per new GGUF plus runtime candidate; no official SR."""
from pathlib import Path
import csv,json,os,shlex,subprocess,threading,time
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'candidate-smokes';out.mkdir(exist_ok=False)
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
cases=[('flash-mm',source,['--flash-attn','--mm-prec','default'])]+[(p.stem,p,[]) for p in sorted((root/'research-notes/generated-models').glob('*.gguf'))]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
env.update(MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',LIBERO_CONFIG_PATH=str(base/'libero-config'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
results=[]
for name,model,flags in cases:
    folder=out/name;folder.mkdir()
    server_cmd=[str(root/'build-cuda/vla-server'),'--bind','tcp://127.0.0.1:15558',str(model)]+flags
    client_cmd=[str(root/'eval/sim/libero/libero_uv/.venv/bin/python'),str(root/'eval/client/run_sim_client_direct.py'),'--arch','smolvla','--vla-addr','tcp://127.0.0.1:15558','--task','libero_object','--task-id','0','--n-episodes','2','--n-action-steps','1','--seed','42','--output-dir',str(folder/'outputs')]
    (folder/'commands.json').write_text(json.dumps(dict(server=server_cmd,client=client_cmd,classification='smoke only',noise='repo defaults, random server RNG',tokenizer='same cached HF tokenizer, offline to freeze cached assets'),indent=2)+'\n')
    (out/'progress.json').write_text(json.dumps(dict(state='running',current=name,completed=len(results),expected=len(cases)),indent=2)+'\n')
    print('SMOKE starting',name,'1 task x 2 episodes',flush=True)
    print('SERVER:',shlex.join(server_cmd),flush=True)
    record=dict(name=name,model=str(model),flags=flags,expected_episodes=2,status='incomplete')
    with (folder/'server.log').open('w') as slog:
        server=subprocess.Popen(server_cmd,cwd=root,env=env,stdout=slog,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while 'ready.' not in (folder/'server.log').read_text():
                if server.poll() is not None:raise RuntimeError('Server exited before readiness')
                if time.monotonic()>deadline:raise TimeoutError('Server readiness timeout')
                time.sleep(.5)
            print('CLIENT:',shlex.join(client_cmd),flush=True)
            with (folder/'client.log').open('w') as clog:
                client=subprocess.Popen(client_cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                def pump(pipe=client.stdout,log=clog,label=name):
                    for line in pipe:
                        log.write(line);log.flush()
                        if line.startswith(('*** Episode','- Episode finished','- Final reward','- Success rate','*** All episodes')):print(label+': '+line.rstrip(),flush=True)
                thread=threading.Thread(target=pump,daemon=True);thread.start()
                try:code=client.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    client.terminate()
                    try:client.wait(timeout=10)
                    except subprocess.TimeoutExpired:client.kill();client.wait()
                    raise
                finally:thread.join(timeout=10)
            record['client_exit']=code
            if code:raise RuntimeError(f'Client exit {code}; inspect client.log')
            path=folder/'outputs/smolvla/libero_object/task_0/episodes.csv'
            with path.open() as f:episodes=list(csv.DictReader(f))
            record['completed_episodes']=len(episodes)
            if len(episodes)!=2 or any(int(r['aborted']) for r in episodes):raise RuntimeError('Missing/aborted smoke episodes')
            record.update(status='complete',successes=sum(int(r['success']) for r in episodes),steps=[int(r['steps']) for r in episodes])
        except Exception as error:
            record['error']=str(error)
            print('SMOKE error',name,str(error),flush=True)
        finally:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()
    results.append(record)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('SMOKE result',name,record['status'],record.get('successes'),'/2',flush=True)
    subprocess.run(['python3',str(base/'update_live_report.py')],check=True)
    with (root/'NOTES_SMOLVLA.md').open('a') as notes:notes.write(f"\nSmoke {name}: {record['status']}, successes={record.get('successes')}/2, steps={record.get('steps')}; logs candidate-smokes/{name}. Không phải screening hoặc official.\n")
(out/'progress.json').write_text(json.dumps(dict(state='complete',completed=len(results),expected=len(cases)),indent=2)+'\n')
