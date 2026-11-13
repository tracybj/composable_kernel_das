"""Verify retained official binaries and every declared compilation input copy."""
from pathlib import Path
import datetime, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
BENCH = Path('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    candidates = [BENCH] + sorted((ROOT / 'compile').glob('build*/previous/' + BENCH.name))
    binaries = {}
    for path in candidates:
        binaries.setdefault(sha(path), []).append(str(path))
    rows = []
    for number in range(9, 18):
        folder = ROOT / f'compile/build{number}'
        completed = json.loads((folder / 'completed.json').read_text())
        pins = json.loads((folder / 'source_sha256.json').read_text())
        assert completed['source_sha256'] == pins
        binary = completed['benchmark_sha256']
        assert binary in binaries, f'Actual official binary unavailable for build{number}'
        source = folder / 'compiled_sources'
        if number == 9:
            source = ROOT / 'compile/build10/previous/source'
        assert source.is_dir()
        for name, digest in pins.items():
            assert sha(source / name) == digest, (number, name)
        rows.append(dict(build=number, benchmark_sha256=binary, actual_binary_paths=binaries[binary],
                         compile_input_count=len(pins), compile_input_source=str(source),
                         every_declared_source_copy_SHA_verified=True, source_manifest_sha256=sha(folder / 'source_sha256.json')))
    report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), ok=True,
                  CPU_only=True, GPU=False, official_builds=len(rows),
                  compile_input_copies_verified=sum(x['compile_input_count'] for x in rows), rows=rows,
                  note='Compiled sources include the protected user overlay; no claim that clean HEAD alone is the frozen measurement binary.')
    (ROOT / 'readonly_reports/binary_source_inventory_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OFFICIAL_BINARY_AND_COMPILE_INPUT_SNAPSHOTS_PASS', len(rows), report['compile_input_copies_verified'], flush=True)


if __name__ == '__main__':
    main()
