"""Extract integer counter records from the four audited HYGPERF captures.

This is a bounded reader for the observed file format, not an xprof UI exporter.
It rejects a different record layout and never treats derived percentages or
aggregate wave-cycle values as kernel wall time. Preserve the original .perf.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, re, struct, zlib


def parse(path):
    data = path.read_bytes()
    assert data[:8] == b'\xe4HYGPERF'
    tables = []
    for match in re.finditer(struct.pack('<Q', 6), data):
        pos = match.start()
        if pos < 24 or pos + 48 > len(data):
            continue
        if struct.unpack_from('<QQ', data, pos + 8) != (pos - 24, 48):
            continue
        obj = zlib.decompressobj()
        strings = obj.decompress(data[pos + 48:])
        assert obj.eof and len(strings) == struct.unpack_from('<Q', data, pos + 32)[0]
        tables.append(strings)
    assert len(tables) == 1
    strings, counters = tables[0], {}
    for match in re.finditer(struct.pack('<Q', 10), data):
        pos = match.start()
        if pos < 24 or pos + 64 > len(data):
            continue
        if struct.unpack_from('<QQ', data, pos + 8) != (pos - 24, 64):
            continue
        dispatch, start, length, value, flags = struct.unpack_from('<QQQQQ', data, pos + 24)
        assert dispatch == 86 and start + length <= len(strings) and flags == 0x90d
        name = strings[start:start + length].decode('ascii')
        assert re.fullmatch(r'[A-Za-z0-9_\[\]]+', name) and name not in counters
        counters[name] = value
    assert len(counters) == 255
    # Independent geometry/work checks: 1024 CTAs, two waves per CTA,
    # and 192 MMAC instructions per wave at K6144.
    assert counters['SQ_WAVES'] == 2048 and counters['SQ_INSTS_MMOP'] == 393216
    return dict(perf_sha256=hashlib.sha256(data).hexdigest(), counters=counters)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture_directory', type=Path)
    args = ap.parse_args()
    root, output = args.capture_directory, {}
    for name in ['both_lds', 'a_reg', 'raw_reg', 'packed_reg']:
        files = list((root / name).glob('*.perf'))
        assert len(files) == 1
        metadata = (root / name / 'results.txt').read_text()
        assert 'gpu-id(7)' in metadata and 'grd(16384, 1, 8)' in metadata and 'wgr(128, 1, 1)' in metadata
        output[name] = dict(**parse(files[0]), dispatch=metadata.splitlines()[0])
    (root / 'uint64_counters.json').write_text(json.dumps(output, indent=2) + '\n')
    selected = ['SQ_INSTS_VMEM', 'SQ_INSTS_LDS', 'SQ_INSTS_MMOP', 'SQ_INSTS_VALU',
                'TCP_TOTAL_READ_sum', 'TCP_TOTAL_CACHE_ACCESSES_sum',
                'TCP_READ_TAGCONFLICT_STALL_CYCLES_sum', 'TCP_TCP_TA_ADDR_STALL_CYCLES_sum',
                'TCP_TCC_READ_REQ_sum',
                'TCC_HIT_sum', 'TCC_MISS_sum', 'TCC_EA_RDREQ_sum',
                'TCC_EA_RDREQ_32B_sum', 'SQ_LEVEL_WAVES_accum']
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['metric', *output])
    for metric in selected:
        writer.writerow([metric, *[row['counters'][metric] for row in output.values()]])
    (root / 'comparison_counters.csv').write_text(buffer.getvalue())
    print(buffer.getvalue())


if __name__ == '__main__':
    main()
