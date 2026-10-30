"""Build the maintained INT8-only cached API and freeze its source identity.

This does not run GPU work. Run the boundary/lifetime gates before the full suite.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    extension = {}
    for path in E.rglob('*'):
        if not path.is_file():
            continue
        if any(part in ['experiments', 'docs', 'tests', 'build', 'results', '__pycache__']
               or part.startswith('build_') for part in path.relative_to(E).parts):
            continue
        if path.suffix in ['.cpp', '.hpp', '.h', '.txt']:
            extension[str(path.relative_to(ROOT))] = digest(path)
    context = ['include/ck_tile/ops/gemm/grouped_gemm_selector.hpp',
               'include/ck_tile/ops/gemm/grouped_gemm_device_args.h',
               'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp']
    return dict(extension=extension,
                existing_public_context_unchanged={name: digest(ROOT / name) for name in context})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=E / 'build/refactor_v121')
    parser.add_argument('--compiler', default='/opt/dtk/bin/aicc')
    parser.add_argument('--arch', default='gfx936')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    assert args.jobs > 0
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    before = sources()
    commands = [
        ['cmake', '-S', str(E / 'cached_weights'), '-B', str(build),
         '-DCMAKE_CXX_COMPILER=' + args.compiler, '-DCMAKE_BUILD_TYPE=Release',
         '-DCK_INT8_CACHED_ARCH=' + args.arch],
        ['cmake', '--build', str(build), '--target',
         'ck_tile_int8_nt_cached_benchmark', '-j' + str(args.jobs)],
    ]
    (build / 'build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for phase, command in zip(['configure', 'compile'], commands):
        with (build / (phase + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (phase, build / (phase + '.log'))
    assert before == sources(), 'Source changed during compilation'
    library = build / 'libck_tile_int8_nt_cached.so'
    exports = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    names = [line.split()[-1] for line in exports.splitlines() if line.split()]
    api = [name for name in names if name.startswith('ck_tile_int8_nt_cached_')]
    assert len(api) == 8, api
    assert not any(name.startswith('ck_tile_hcu_grouped_gemm_')
                   or name == 'ck_tile_grouped_int8_nt_launch' for name in names)
    (build / 'sources.json').write_text(json.dumps(before, indent=2) + '\n')
    identity = dict(symbols=api, library_sha256=digest(library),
                    benchmark_sha256=digest(build / 'libck_tile_int8_nt_cached_benchmark.so'))
    (build / 'exported_api_symbols.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(dict(status='BUILD_PASS_GPU_NOT_RUN', build=str(build), **identity), indent=2))


if __name__ == '__main__':
    main()
