#!/usr/bin/env python3
"""Read-only closing check against actual hymgr service interruptions."""
import datetime,json,re,subprocess
from pathlib import Path
import journalview
import run as r

p=r.ROOT
dt=datetime.datetime
td=datetime.timedelta

def main():
    output=subprocess.check_output(['dmesg','--time-format','iso'],text=True)
    lines=[line for line in output.splitlines() if 'hymgr' in line and 'register hyuds' in line]
    (p/'kernel_environment_final.log').write_text('\n'.join(lines)+'\n')
    events=[dict(at=dt.fromisoformat(line.split()[0].replace(',','.')).astimezone(datetime.timezone.utc).isoformat(),line=line) for line in lines if 'unregister hyuds' in line]
    observed=dt.now(datetime.timezone.utc)
    raw,good=journalview.read()
    def end(row):return dt.fromisoformat(row['at'])+td(seconds=row['seconds'])
    jobs=[]
    for key,row in good.items():
        for event in events:
            if dt.fromisoformat(row['at'])<=dt.fromisoformat(event['at'])<=end(row):
                jobs.append(dict(key=key,raw_key=row['key'],event=event))
    pairs=[]
    for key,row in good.items():
        after=None
        if key.startswith(('perf_cache_','expand_cache_')) and key.endswith('_base'):after=key[:-5]+'_noslc'
        if key.startswith(('smallperf_','smallexpand_','squarediag_')) and key.endswith('_before'):after=key[:-7]+'_after'
        if key.startswith('walk_cache_') and key.endswith('_noslc'):after=key+'_row'
        if key.startswith('nativeref_') and key.endswith('_noslc'):after=key[:-6]+'_blas'
        if after is None or after not in good:continue
        for event in events:
            if dt.fromisoformat(row['at'])<=dt.fromisoformat(event['at'])<=end(good[after]):
                pairs.append(dict(before=key,after=after,event=event))
    result=dict(observed_at=observed.isoformat(),accepted_jobs=len(good),last_accepted_end=max(end(row) for row in good.values()).isoformat(),all_accepted_jobs_finished_before_observation=all(end(row)<observed for row in good.values()),accepted_job_service_interruptions=jobs,paired_service_interruptions=pairs,service_unregister_events=events,source='kernel_environment_final.log',limitation='Read-only retrospective observation, not a future stability guarantee. Run after campaign process exits.')
    (p/'final_environment_observation.json').write_text(json.dumps(result,indent=2))
    assert result['all_accepted_jobs_finished_before_observation'] and not jobs and not pairs,result
    print('FINAL_ENVIRONMENT_OBSERVATION_PASS',len(good),'accepted jobs; no service interruption inside accepted jobs or pairs')

if __name__=='__main__':main()
