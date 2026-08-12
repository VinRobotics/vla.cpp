#!/usr/bin/env python3
"""Checks the ctypes structs against include/vla.h.

A field added on one side and not the other silently misreads every value after
it. Set VLA_LIBRARY to also load the library and check its ABI version.
"""

import ctypes
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "bindings", "python"))

from vla_cpp import _ffi  # noqa: E402

HEADER = os.path.join(ROOT, "include", "vla.h")

C_TO_CTYPES = {
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}


def header_struct_fields(text, name):
    """Field (name, c_type) pairs of `typedef struct { ... } name;`."""
    # [^{}] so the body cannot swallow the structs in between.
    m = re.search(r"typedef struct \{([^{}]*)\}\s*" + name + r"\s*;", text, re.S)
    assert m, f"{name} not found in vla.h"
    fields = []
    for line in m.group(1).splitlines():
        line = re.sub(r"/\*.*?\*/", "", line)
        line = line.split("///")[0].split("//")[0].strip()
        if not line or not line.endswith(";"):
            continue
        decl = line[:-1].strip()
        parts = decl.split()
        if len(parts) < 2:
            continue
        ctype, fname = parts[0], parts[-1].lstrip("*")
        fields.append((fname, ctype))
    return fields


def check_struct(text, c_name, py_struct, skip_types=()):
    hdr = header_struct_fields(text, c_name)
    py = [(n, t) for n, t in py_struct._fields_]
    assert len(hdr) == len(py), (
        f"{c_name}: vla.h has {len(hdr)} fields, python has {len(py)}\n"
        f"  header: {[n for n, _ in hdr]}\n  python: {[n for n, _ in py]}"
    )
    for (hn, ht), (pn, pt) in zip(hdr, py):
        assert hn == pn, f"{c_name}: field order differs, {hn!r} vs {pn!r}"
        if ht in skip_types:
            continue
        want = C_TO_CTYPES.get(ht)
        if want is not None:
            assert pt is want, f"{c_name}.{hn}: header {ht}, python {pt}"
    print(f"  {c_name}: {len(hdr)} fields match")


def main():
    text = open(HEADER).read()

    m = re.search(r"#define VLA_ABI_VERSION\s+(\d+)", text)
    assert m, "VLA_ABI_VERSION not found"
    assert int(m.group(1)) == _ffi.ABI_VERSION, (
        f"vla.h ABI {m.group(1)}, _ffi.ABI_VERSION {_ffi.ABI_VERSION}")
    print(f"  ABI version {_ffi.ABI_VERSION} matches")

    check_struct(text, "vla_config", _ffi.Config)
    check_struct(text, "vla_stats", _ffi.Stats)
    # Pointer fields differ in spelling between the two; only order is checked.
    check_struct(text, "vla_image", _ffi.Image, skip_types=("void",))
    check_struct(text, "vla_inputs", _ffi.Inputs,
                 skip_types=("vla_image", "float", "int32_t"))

    for name in ("VLA_OK", "VLA_ERR_ARG", "VLA_ERR_PREDICT", "VLA_ERR_EXCEPTION"):
        assert name in text, f"{name} missing from vla.h"
    assert _ffi.OK == 0 and _ffi.ERR_ARG == -1
    assert _ffi.ERR_PREDICT == -2 and _ffi.ERR_EXCEPTION == -3
    print("  status codes match")

    if os.environ.get("VLA_LIBRARY"):
        lib = _ffi.load_library()
        assert lib.vla_abi_version() == _ffi.ABI_VERSION
        print("  libvla loads and reports a matching ABI")

    print("bindings: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
