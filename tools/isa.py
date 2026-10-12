"""See what the Intel GPU compiler actually produced for an OpenCL kernel.

    python tools/isa.py kernels/kernels.cl matmul_sg [extra build options]

Builds the file on the first Intel OpenCL device (pyopencl), pulls the machine code of one kernel out of the
program binary (a ZeBin ELF; the code is in the `.text.<kernel>` section) and disassembles it with Intel's
IGA library, which ships in the graphics driver store (`iga64.dll`). Writes `isa_<kernel>.txt` and prints
the first lines.

This is how the matmul was fixed: the disassembly showed the B loads issued right before the FMAs that
use them (every k step stalled on a cache round trip) and long runs of 64-bit scalar address arithmetic
around each load, neither of which is visible from the OpenCL source. Useful checks:

    grep -c "mad (16"  isa_matmul_sg.txt     # FMA instructions
    grep -c scratch    isa_matmul_sg.txt     # register spills (should be 0)
    grep "send"        isa_matmul_sg.txt     # the loads and stores, with their message shapes

Needs `pip install pyopencl` (not a dependency of the plugin) and an Intel GPU driver. IGA's C API is
used through ctypes with the signatures below; if a driver update changes them, `iga_context_disassemble`
will return a nonzero status instead of text.

IGA platform id (`iga_gen_t`): 0x1000002 decodes Arc 140T (Xe-LPG); override with IGA_GEN=0x... .
"""

import ctypes
import glob
import os
import struct
import sys

import pyopencl as cl

src_file, kernel = sys.argv[1], sys.argv[2]
extra = " ".join(sys.argv[3:])
gen = int(os.environ.get("IGA_GEN", "0x1000002"), 0)

device = [d for p in cl.get_platforms() for d in p.get_devices() if "Intel" in d.name][0]
context = cl.Context([device])
program = cl.Program(context, open(src_file, encoding="utf-8").read()).build(
    options="-cl-intel-greater-than-4GB-buffer-required " + extra
)
binary = program.binaries[0]

# --- find the kernel's code in the ELF
assert binary[:4] == b"\x7fELF", "unexpected program binary format"
shoff = struct.unpack_from("<Q", binary, 0x28)[0]
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", binary, 0x3A)
sections = [struct.unpack_from("<IIQQQQ", binary, shoff + i * shentsize) for i in range(shnum)]
names_at = sections[shstrndx][4]


def section_name(offset):
    return binary[names_at + offset : binary.index(b"\0", names_at + offset)].decode()


code = next(
    (binary[off : off + size] for name, _, _, _, off, size in sections if section_name(name) == ".text." + kernel),
    None,
)
if code is None:
    sys.exit(f"no kernel {kernel!r}; sections: {[section_name(s[0]) for s in sections]}")

# --- disassemble
dlls = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*\iga64.dll")
if not dlls:
    sys.exit("iga64.dll not found in the driver store")
iga = ctypes.WinDLL(dlls[-1])


class ContextOptions(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_size_t), ("gen", ctypes.c_int)]


class DisassembleOptions(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_size_t),
        ("formatting", ctypes.c_uint32),
        ("decoding", ctypes.c_uint32),
        ("syntax", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


ctx = ctypes.c_void_p()
options = ContextOptions(ctypes.sizeof(ContextOptions), gen)
if iga.iga_context_create(ctypes.byref(options), ctypes.byref(ctx)):
    sys.exit("iga_context_create failed (unknown platform id?)")
text = ctypes.c_char_p()
buf = ctypes.create_string_buffer(code, len(code))
for cb in (16, 20, 24, 32):  # the options struct size differs between IGA versions
    opts = DisassembleOptions(cb, 0, 0, 0, 0)
    status = iga.iga_context_disassemble(ctx, ctypes.byref(opts), buf, len(code), None, None, ctypes.byref(text))
    if status == 0 and text.value:
        out = f"isa_{kernel}.txt"
        listing = text.value.decode(errors="replace")
        open(out, "w").write(listing)
        print(f"{len(code)} bytes of code -> {out}")
        print("\n".join(listing.splitlines()[:40]))
        break
else:
    sys.exit("iga_context_disassemble failed for platform 0x%x" % gen)
