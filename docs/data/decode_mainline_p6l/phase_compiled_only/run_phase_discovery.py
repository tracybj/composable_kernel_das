"""One finite serial phase campaign; preserve errors without automatic restart."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
SELF=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())
def main():
    with (ROOT/'phase_controller.lock').open('a') as lock,(ROOT/'final_controller.lock').open('a') as final:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(final,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def event(**kw):
            item=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),script_sha256=SELF,**kw)
            p=ROOT/'phase_controller_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(item,indent=2)+'\n');tmp.replace(p)
            with (ROOT/'phase_controller_events.jsonl').open('a') as out:out.write(json.dumps(item)+'\n');out.flush();os.fsync(out.fileno())
        def command(script,stage=None,gpu=False):
            assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==SELF
            session=read(ROOT/'session.json');assert time.time()<session['deadline_epoch']
            if gpu:
                assert time.time()+60<session['gpu_deadline_epoch']
                assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json','user_pause.json'])
            argv=[sys.executable,'-B','-u',str(ROOT/script)]+([stage] if stage else [])
            event(state='running',script=script,stage=stage,command=argv)
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL);rc=child.wait()
            event(state='passed' if rc==0 else 'failed',script=script,stage=stage,child_pid=child.pid,rc=rc)
            print('PHASE_CONTROLLER_STAGE',script,stage,'rc',rc,flush=True)
            if rc:raise RuntimeError(f'{script}/{stage} failed; no automatic retry')
        try:
            for stage in ['freeze','qualify']:command('phase_campaign.py',stage,True)
            command('audit_phase_qualification.py')
            command('phase_campaign.py','discover',True)
            command('analyze_phase_discovery.py')
            for stage in ['freeze','capture','analyze','deliver']:command('phase_profiles_campaign.py',stage,stage in ['freeze','capture'])
            event(state='complete',note='Discovery and profiles complete; no adoption or Goal completion without whole-domain seven-block confirmation')
            print('PHASE_DISCOVERY_AND_PROFILES_COMPLETE',flush=True)
        except BaseException as error:
            event(state='stopped',error=str(error),error_type=type(error).__name__,automatic_retry=False);raise

if __name__=='__main__':main()
