#!/usr/bin/env python3
"""Check shape-only INT8 NT selection against recorded instances (CPU only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

QUERY = r'''
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <cstdint>
#include <iostream>
int main() {
    using namespace ck_tile;
    std::int64_t m, n, k;
    int g;
    while (std::cin >> m >> n >> k >> g) {
        GroupedGemmProblem p{};
        p.arch = GroupedGemmArch::gfx936;
        p.data_type = GroupedGemmDataType::int8_int32;
        p.layout = GroupedGemmLayout::nt;
        p.m = GroupedGemmDimensionSummary::Common(m);
        p.n = GroupedGemmDimensionSummary::Common(n);
        p.k = GroupedGemmDimensionSummary::Common(k);
        p.group_count = g;
        std::cout << static_cast<unsigned>(select_grouped_gemm_candidate(p).instance_id) << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    with args.results.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows, 'Empty results CSV'
    for row in rows:
        assert row['dtype'] == 'w8a8_int8' and row['layout'] == 'NT'
        assert row['output dtype'] == 'int32' and row['scale'] == 'none'

    def payload(records):
        # Deliberately expose only the operator dimensions to the C++ selector.
        return ''.join(f"{r['M']} {r['N']} {r['K']} {r['E']}\n" for r in records)

    with tempfile.TemporaryDirectory(prefix='int8_shape_selector_') as directory:
        temp = Path(directory)
        source, exe = temp/'query.cpp', temp/'query'
        source.write_text(QUERY)
        subprocess.run([args.cxx, '-std=c++17', '-O2', '-I'+str(root/'include'),
                        str(source), '-o', str(exe)], check=True)

        def query(records):
            result = subprocess.run([str(exe)], input=payload(records), text=True,
                                    capture_output=True, check=True)
            return [int(x) for x in result.stdout.splitlines()]

        selected = query(rows)
        assert selected == [int(r['instance_id']) for r in rows], 'Recorded selection changed'
        relabelled = [dict(r, ID=f'arbitrary_{i}', phase='unused_label')
                      for i, r in enumerate(reversed(rows))]
        assert query(relabelled) == list(reversed(selected)), 'Selection depends on ordering/labels'
        for group_count in (0, 65536):
            invalid = dict(rows[0], E=str(group_count))
            assert query([invalid]) == [0], 'Invalid group count accepted'

    by_shape = {}
    for row, instance in zip(rows, selected):
        shape = tuple(int(row[k]) for k in ('M','N','K','E'))
        assert by_shape.setdefault(shape, instance) == instance, 'Duplicate shape selected differently'
    audit = dict(status='PASS', rows=len(rows), unique_shapes=len(by_shape),
                 instance_count=len(set(selected)), shape_fields=['M','N','K','G'],
                 formal_instances_match=True, relabelled_reversed_match=True,
                 invalid_group_counts_rejected=True, gpu_work_performed=False,
                 reference_csv_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
