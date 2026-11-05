"""Model asynchronous B writes versus both halves' final MMAC uses.

The model proves slot ownership/order, not compiled instruction scheduling or
GPU performance. The A publication/retirement protocol is unchanged.
"""


def check(stages, single, early):
    buffers = {}
    pending = {0: 0} if stages else {}
    retired = {0: True, 1: True}
    seen = []
    checks = 0

    def issue(slot, stage):
        nonlocal checks
        assert retired[slot] and slot not in pending
        pending[slot] = stage
        checks += 1

    for stage in range(stages):
        slot = 0 if single else stage % 2
        # VMEM wait completes all previously issued B loads before low use.
        buffers.update(pending)
        pending.clear()
        assert buffers[slot] == stage
        retired[slot] = False
        if not single and early and stage + 1 < stages:
            issue(slot ^ 1, stage + 1)
        seen.append((stage, "low"))
        if not single and not early and stage + 1 < stages:
            issue(slot ^ 1, stage + 1)
        assert buffers[slot] == stage  # No overwrite between low and high.
        seen.append((stage, "high"))
        retired[slot] = True
        if single and stage + 1 < stages:
            issue(0, stage + 1)
        # Current A's final use is complete before the unchanged stage barrier.
        checks += 1
    assert not pending
    assert seen == [(stage, half) for stage in range(stages) for half in ("low", "high")]
    return checks


def main():
    checks = sum(check(n, single, early)
                 for n in range(66)
                 for single in (False, True)
                 for early in (False, True))
    print("SINGLE_B_PROTOCOL_PASS checks=", checks)


if __name__ == "__main__":
    main()
