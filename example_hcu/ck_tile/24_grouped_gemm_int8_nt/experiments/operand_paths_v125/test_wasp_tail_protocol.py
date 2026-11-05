"""Model the barrier epochs and LDS slot ownership of masked N consumers.

This is a protocol/layout proof, not a substitute for GPU validation of the
compiled producer/consumer implementation.
"""


def producer(stages):
    trace = [[("write", 0)]]  # Between entry and initial publication.
    for stage in range(1, stages):
        trace.append([("write", stage)])
    if stages:
        trace.append([])  # Final retirement has no next write.
    return trace + [[]]  # Common final sync.


def active_consumer(stages):
    return [[]] + [[("read", stage)] for stage in range(stages)] + [[]]


def inactive_consumer(stages):
    # Initial ready + one epoch per stage; common final sync is outside branch.
    return [[] for _ in range(stages + 2)]


def main():
    checks = 0
    for stages in range(66):
        p, c, idle = producer(stages), active_consumer(stages), inactive_consumer(stages)
        assert len(p) == len(c) == len(idle) == stages + 2
        slots = {}
        for epoch, (writes, reads) in enumerate(zip(p, c)):
            # Reads and writes in this segment may overlap in either order.
            for op, stage in reads:
                assert op == "read" and slots[stage % 2] == stage
                assert all(other % 2 != stage % 2 for _, other in writes)
            for op, stage in writes:
                assert op == "write"
                slots[stage % 2] = stage
            checks += 1
    for n in [256, 512, 768, 1536, 1792, 2048, 3072, 4096, 5120, 6144, 7168]:
        columns = []
        for origin in range(0, n, 384):
            for consumer in range(12):
                begin = origin + consumer * 32
                if begin < n:
                    assert begin + 32 <= n
                    columns.extend(range(begin, begin + 32))
                checks += 1
        assert columns == list(range(n))
    print("WASP_TAIL_PROTOCOL_PASS checks=", checks)


if __name__ == "__main__":
    main()
