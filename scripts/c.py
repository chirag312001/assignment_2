#!/usr/bin/env python3
# check_core_attrs_verbose.py
# Run with: ~/gem5/build/X86/gem5.opt check_core_attrs_verbose.py

from m5.objects import DerivO3CPU

def main():
    c = DerivO3CPU()
    attrs = dir(c)

    # patterns to look for (case-insensitive)
    patterns = [max_insts_any_thread]

    matched = []
    for a in attrs:
        al = a.lower()
        if any(p in al for p in patterns):
            matched.append(a)

    print("DerivO3CPU attributes matched by patterns ({}):\n".format(len(matched)))
    for a in matched:
        try:
            v = getattr(c, a)
        except Exception:
            v = "<unreadable>"
        print(f" - {a}: {v}")

    # explicit checks for common names
    print("\nExplicit existence/value checks:")
    common = [
        'numIQEntries', 'iqEntries', 'issueQueueEntries',
        'numROBEntries', 'LQEntries', 'SQEntries',
        'numPhysIntRegs', 'numPhysFloatRegs',
        'fetchWidth', 'issueWidth', 'commitWidth',
        'branchPred'
    ]
    for name in common:
        present = hasattr(c, name)
        if present:
            try:
                val = getattr(c, name)
            except Exception:
                val = "<unreadable>"
            print(f" - {name}: PRESENT (value={val})")
        else:
            print(f" - {name}: MISSING")

if __name__ == "__m5_main__":
    main()
