#!/usr/bin/env python3
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Line filter that hides every fragment of the values of the given env vars.

Usage: some-command 2>&1 | python3 -I redact.py VAR [VAR ...]

Any substring of length >= 6 of a named variable's value is replaced, and so
are its prefixes and suffixes of length >= 4. This covers AWF's "key prefix"
debug lines and masked forms such as "sk-...abcd". The values are read from
this process's environment and are never written out.
"""

import os
import sys


def fragments(value: str) -> list[str]:
    out = set()
    n = len(value)
    for length in range(4, n + 1):
        out.add(value[:length])
        out.add(value[n - length :])
    for length in range(6, n + 1):
        for start in range(0, n - length + 1):
            out.add(value[start : start + length])
    return sorted(out, key=len, reverse=True)


def main() -> None:
    frags: list[str] = []
    for name in sys.argv[1:]:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            frags.extend(fragments(value))
    frags = sorted(set(frags), key=len, reverse=True)
    for line in sys.stdin:
        for frag in frags:
            if frag in line:
                line = line.replace(frag, "[REDACTED]")
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
