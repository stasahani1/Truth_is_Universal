#!/usr/bin/env python3
"""Patch PEP-701 f-strings (nested same-type quotes, py3.12-only) down to py3.10 syntax.
For each file: py_compile until clean; on each SyntaxError line, rewrite double quotes
that occur INSIDE {...} expressions of an f"..." string to single quotes. Behavior-neutral."""
import py_compile
import re
import sys


def fix_line(s):
    out = []
    i, n = 0, len(s)
    in_f = False      # inside an f"..." string
    depth = 0         # {...} nesting depth inside the f-string
    while i < n:
        c = s[i]
        if not in_f:
            m = re.match(r'[fF][rR]?"', s[i:])
            if m:
                out.append(s[i:i + m.end()])
                i += m.end()
                in_f = True
                depth = 0
                continue
            out.append(c)
            i += 1
        else:
            if c == "{" and i + 1 < n and s[i + 1] == "{":
                out.append("{{"); i += 2; continue
            if c == "}" and i + 1 < n and s[i + 1] == "}":
                out.append("}}"); i += 2; continue
            if c == "{":
                depth += 1; out.append(c); i += 1; continue
            if c == "}" and depth > 0:
                depth -= 1; out.append(c); i += 1; continue
            if c == '"':
                if depth > 0:
                    out.append("'")   # inner quote inside {expr} -> single
                else:
                    out.append(c)     # closing quote of the f-string
                    in_f = False
                i += 1
                continue
            out.append(c); i += 1
    return "".join(out)


def main(paths):
    for path in paths:
        for _ in range(50):
            try:
                py_compile.compile(path, doraise=True)
                print(f"{path}: CLEAN")
                break
            except py_compile.PyCompileError as e:
                m = re.search(r"line (\d+)", str(e))
                if not m:
                    print(f"{path}: UNPARSEABLE ERROR: {e}"); sys.exit(1)
                ln = int(m.group(1))
                lines = open(path).readlines()
                orig = lines[ln - 1]
                fixed = fix_line(orig)
                if fixed == orig:
                    print(f"{path}:{ln}: CANNOT AUTO-FIX: {orig.strip()[:160]}")
                    sys.exit(1)
                lines[ln - 1] = fixed
                open(path, "w").writelines(lines)
                print(f"{path}:{ln}: {orig.strip()[:100]}")
                print(f"      -> {fixed.strip()[:100]}")


if __name__ == "__main__":
    main(sys.argv[1:])
