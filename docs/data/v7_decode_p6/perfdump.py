#!/usr/bin/env python3
# Decode xprof HYGPERF .perf: dump every (name, value) counter record per dispatch section.
# Layout inferred from the files (not documented): after the 0x98-byte header, a zlib
# block (code object), then records [type u64][self_off u64][len u64][8 u64][payload...];
# type 0x0a payload = (name_off, name_len, value, ?) into the zlib'd name table at the end.
import sys, struct, zlib

def zblocks(d):
    out, i = [], 0
    while True:
        j = d.find(b'\x78\x9c', i)
        if j < 0:
            return out
        try:
            o = zlib.decompressobj()
            u = o.decompress(d[j:])
            used = len(d) - j - len(o.unused_data)
            out.append((j, used, u))
            i = j + used
        except Exception:
            i = j + 2

def parse(path):
    d = open(path, 'rb').read()
    zb = zblocks(d)
    names = zb[-1][2]
    rec_start = zb[0][0] + zb[0][1]
    rec_end = zb[-1][0]
    i = rec_start
    rows, others = [], []
    while i + 32 <= rec_end:
        typ, off, ln, _ = struct.unpack_from('<4Q', d, i)
        if ln == 0 or ln > rec_end - i:
            i += 8
            continue
        body = d[i + 32:i + ln]
        if typ == 0x0a and len(body) >= 24:
            no, nl, val = struct.unpack_from('<3Q', body, 0)
            nm = names[no:no + nl].decode('ascii', 'replace')
            rows.append((nm, val))
        else:
            others.append((typ, ln, body[:64].hex()))
        i += ln
    return rows, others

def dispatches(path):
    # Counters repeat per dispatch in the same order; split on the first name's recurrence.
    rows, _ = parse(path)
    out, cur = [], {}
    for nm, v in rows:
        if nm in cur:
            out.append(cur)
            cur = {}
        cur[nm] = v
    if cur:
        out.append(cur)
    return out

if __name__ == '__main__':
    for p in sys.argv[1:]:
        ds = dispatches(p)
        print('#', p, 'dispatches', len(ds))
        last = ds[-1]
        for nm, v in last.items():
            print('%-40s %d' % (nm, v))
