"""Replay all exact policy-reader cases and public masks; no HIP link or GPU."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
PREFIX = 'CK_TILE_GROUPED_GEMM_'


def main():
    started = time.monotonic()
    fixture = ROOT / 'drafts/conditional_n128_wave4/cpu_policy'
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    linked = subprocess.check_output(['ldd', str(D / 'policy_equivalence')], text=True)
    assert not any(x in linked.lower() for x in ['libhsa', 'libamdhip', 'libgalaxyhip', 'libhipblas'])
    (D / 'ldd.txt').write_text(linked)
    counts = []
    for filename, expected in [('cases.json', 175389), ('public_disable_cases.json', 147456)]:
        cases = json.loads((fixture / filename).read_text())
        assert len(cases) == expected
        lines = []
        for index, case in enumerate(cases):
            shape, flags = case['shape'], case['env']
            props = case.get('properties', {})
            disabled = case.get('public_disable_mask', props.get('disabled', 0))
            values = [index, props.get('arch', 1), props.get('type', 2), props.get('layout', 2),
                      shape['M'], shape['N'], shape['K'], shape['G'], props.get('kind', 0),
                      props.get('empty', 0), props.get('ragged', 0), disabled, 0, len(flags)]
            lines.append(' '.join(map(str, values)) + ''.join(' ' + k + ' ' + (v or '~') for k, v in sorted(flags.items())))
        result = subprocess.run([str(D / 'policy_equivalence')], input='\n'.join(lines) + '\n',
                                text=True, capture_output=True, env=env, timeout=180)
        (D / (filename + '.run.log')).write_text(result.stdout + result.stderr)
        assert result.returncode == 0
        outputs = result.stdout.splitlines()
        assert len(outputs) == len(cases)
        for index, line in enumerate(outputs):
            values = list(map(int, line.split()))
            assert len(values) == 7 and values[0] == index
            assert values[1:4] == values[4:7], (filename, cases[index], values)
        counts.append(dict(fixture=filename, cases=len(cases), all_status_instance_reason_equal=True,
                           fixture_sha256=hashlib.sha256((fixture / filename).read_bytes()).hexdigest()))
        print('EXACT_POLICY_REPLAY_PASS', filename, len(cases), flush=True)
    report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), ok=True,
                  CPU_only=True, GPU=False, no_HIP_link=True, groups=counts,
                  cases=sum(x['cases'] for x in counts), seconds=time.monotonic() - started,
                  source_sha256=hashlib.sha256(Path('/workspace/composable_kernel/example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp').read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256((D / 'policy_equivalence').read_bytes()).hexdigest(),
                  not_GPU_correctness_or_performance=True)
    (D / 'policy_equivalence.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
