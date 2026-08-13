#!/usr/bin/env python3
# Copyright 2026 VinRobotics - Apache-2.0
#
# Applies the structural half of the house style to C/C++ sources: one
# statement per line, no brace-wrapped one-liner blocks, no statement stapled
# to the end of an if/else.
#
# Whitespace only. Strings, character literals, comments and preprocessor lines
# are copied through untouched, and the splitter never runs inside parentheses,
# so a for-header's semicolons stay put.
#
#   scripts/restyle.py src/models/vla_adapter.cpp ...

import re
import sys

def spans_to_skip(line):
    """Character ranges of string/char literals and comments in `line`."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in '"\'':
            j = i + 1
            while j < n:
                if line[j] == '\\':
                    j += 2
                    continue
                if line[j] == c:
                    break
                j += 1
            out.append((i, min(j + 1, n)))
            i = j + 1
        elif line.startswith('//', i):
            out.append((i, n))
            break
        elif line.startswith('/*', i):
            j = line.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append((i, j))
            i = j
        else:
            i += 1
    return out

def code_positions(line):
    """Indices of `line` that are real code."""
    skip = spans_to_skip(line)
    return [i for i in range(len(line)) if not any(a <= i < b for a, b in skip)]

def split_statements(body, indent):
    """`a; b; c` -> one statement per line, ignoring semicolons inside parens."""
    parts, depth, cur = [], 0, ''
    live = set(code_positions(body))
    for i, ch in enumerate(body):
        if i in live:
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == ';' and depth == 0:
                parts.append(cur.strip() + ';')
                cur = ''
                continue
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return [indent + p for p in parts if p.strip(';').strip()]

ONE_LINE_BLOCK = re.compile(r'^(?P<indent>\s*)(?P<head>.*?\{)\s*(?P<body>[^{}]*?;)\s*\}\s*$')
CONTROL_HEAD   = re.compile(r'^(?P<indent>\s*)(?:\}\s*else\s+)?(?:if|for|while)\s*\(')

def control_body(line):
    """Split `if/for/while (...) stmt;` into (head, body); None if it is not one."""
    m = CONTROL_HEAD.match(line)
    if not m:
        return None
    live = set(code_positions(line))
    i = line.index('(', m.end() - 1)
    depth = 0
    for j in range(i, len(line)):
        if j not in live:
            continue
        if line[j] == '(':
            depth += 1
        elif line[j] == ')':
            depth -= 1
            if depth == 0:
                head, body = line[:j + 1], line[j + 1:].strip()
                if body and body.endswith(';') and '{' not in body and ';' not in body[:-1]:
                    return head, body
                return None
    return None

def restyle_line(line):
    if line.lstrip().startswith('#'):
        return [line]
    if '"' in line or "'" in line:
        # A literal containing a brace or semicolon would confuse the splitters.
        lit = ''.join(line[a:b] for a, b in spans_to_skip(line))
        if any(c in lit for c in '{};'):
            return [line]

    m = ONE_LINE_BLOCK.match(line)
    if m and 'namespace' not in m.group('head'):
        inner = m.group('indent') + '    '
        stmts = split_statements(m.group('body'), inner)
        if len(stmts) >= 1:
            return [m.group('indent') + m.group('head')] + stmts + [m.group('indent') + '}']

    m = re.match(r'^(?P<indent>\s*)(?:\}\s*)?else\s+(?P<body>[^{};]*?;)\s*$', line)
    if m and not re.match(r'^\s*(?:\}\s*)?else\s+if\b', line):
        head = line[:line.index('else') + 4]
        return [head.rstrip(), m.group('indent') + '    ' + m.group('body').strip()]

    cb = control_body(line)
    if cb:
        head, body = cb
        indent = re.match(r'\s*', line).group(0)
        return [head, indent + '    ' + body]

    return [line]

def restyle(text):
    total = 0
    for _ in range(6):
        out, changed = [], 0
        for line in text.split('\n'):
            new = restyle_line(line)
            if new != [line]:
                changed += 1
            out.extend(new)
        text = '\n'.join(out)
        total += changed
        if not changed:
            break
    return text, total

if __name__ == '__main__':
    total = 0
    for path in sys.argv[1:]:
        src = open(path).read()
        dst, n = restyle(src)
        if n:
            open(path, 'w').write(dst)
        print(f'{path}: {n} lines expanded')
        total += n
    print(f'total {total}')
