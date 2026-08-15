#!/usr/bin/env python3
# Copyright 2026 VinRobotics - Apache-2.0
#
# Removes the spaces around binary operators, per the house style. Whitespace
# only, and deliberately conservative: it edits an operator only when both
# sides are unambiguously values.
#
# Left untouched on purpose:
#   *  &     unless one side is a literal, a ')' / ']', or a '.'/'->' member,
#            because `ggml_tensor * t` is a declaration, not a product
#   <  >     template brackets and includes are indistinguishable here
#   =        `a = b` reads better spaced outside a for-header
#   any operator whose tightening would merge two tokens (`a - -b`)
#
# Strings, character literals, comments and preprocessor lines are copied
# through untouched.

import re
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from restyle import spans_to_skip

VALUE_END   = re.compile(r'(?:[A-Za-z_]\w*|\d[\w.]*|[)\]])$')
VALUE_START = re.compile(r'^(?:[A-Za-z_]\w*|\d|[(\[])')
MEMBER      = re.compile(r'(?:\.\s*[A-Za-z_]\w*|->\s*[A-Za-z_]\w*|\)|\]|\d)$')

OPS = ['+', '-', '/', '%']

def tighten(line):
    if line.lstrip().startswith('#'):
        return line

    skip = spans_to_skip(line)
    live = lambda i: not any(a <= i < b for a, b in skip)

    out = line
    for _ in range(40):
        changed = False
        for m in re.finditer(r'(?<=\S) (' + '|'.join(re.escape(o) for o in OPS + ['*']) + r') (?=\S)', out):
            i, op = m.start(), m.group(1)
            if not live(i):
                continue
            left, right = out[:i], out[i + 3:]
            if not VALUE_END.search(left) or not VALUE_START.match(right):
                continue
            # `a * b` is only a product when one side is plainly a value.
            if op == '*' and not (MEMBER.search(left) or right[0].isdigit()):
                continue
            # never let the operator glue onto a neighbour
            if left[-1] == op or right[0] in '=&|<>+-' or right[0] == op:
                continue
            out = left + op + right
            skip = spans_to_skip(out)
            changed = True
            break
        if not changed:
            break
    return out

if __name__ == '__main__':
    total = 0
    for path in sys.argv[1:]:
        src = open(path).read()
        dst = '\n'.join(tighten(l) for l in src.split('\n'))
        n = sum(1 for a, b in zip(src.split('\n'), dst.split('\n')) if a != b)
        if n:
            open(path, 'w').write(dst)
        total += n
    print(f'tightened {total} lines')
