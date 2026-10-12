"""Build an OpenCL kernel file on the Arc, extract the machine code of one kernel, disassemble with IGA.
usage: isa.py kernels.cl kernel_name [build options...]"""
import ctypes, glob, struct, sys
import pyopencl as cl

src_file, kname, opts = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
dev = [d for p in cl.get_platforms() for d in p.get_devices() if "Intel" in d.name][0]
ctx = cl.Context([dev])
prog = cl.Program(ctx, open(src_file, encoding="utf-8").read()).build(options="-cl-intel-greater-than-4GB-buffer-required " + opts)
binary = prog.binaries[0]
open("prog.bin", "wb").write(binary)

# ELF64 section lookup
assert binary[:4] == b"\x7fELF"
shoff = struct.unpack_from("<Q", binary, 0x28)[0]
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", binary, 0x3A)
secs = []
for i in range(shnum):
    name, typ, flags, addr, off, size = struct.unpack_from("<IIQQQQ", binary, shoff + i * shentsize)
    secs.append((name, typ, off, size))
stroff = secs[shstrndx][2]
def secname(n):
    e = binary.index(b"\0", stroff + n); return binary[stroff + n:e].decode()
text = None
for name, typ, off, size in secs:
    nm = secname(name)
    if nm == ".text." + kname:
        text = binary[off:off + size]
print("kernel bytes:", None if text is None else len(text), [secname(s[0]) for s in secs if secname(s[0]).startswith(".text")])

p = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*\iga64.dll")[-1]
iga = ctypes.WinDLL(p)
LABEL_CB = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p)
def disasm(gen):
    class Opts(ctypes.Structure): _fields_ = [("cb", ctypes.c_size_t), ("gen", ctypes.c_int)]
    ctxp = ctypes.c_void_p()
    o = Opts(ctypes.sizeof(Opts), gen)
    rc = iga.iga_context_create(ctypes.byref(o), ctypes.byref(ctxp))
    if rc: return f"create rc={rc}"
    out = ctypes.c_char_p()
    buf = ctypes.create_string_buffer(text, len(text))
    iga.iga_context_disassemble.restype = ctypes.c_int
    class DOpts(ctypes.Structure): _fields_ = [("cb", ctypes.c_size_t), ("formatting", ctypes.c_uint32), ("decoding", ctypes.c_uint32), ("syntax", ctypes.c_uint32), ("pad", ctypes.c_uint32)]
    last = None
    for cbv in (16, 20, 24, 32, 8):
        d = DOpts(cbv, 0, 0, 0, 0)
        rc = iga.iga_context_disassemble(ctxp, ctypes.byref(d), buf, len(text), None, None, ctypes.byref(out))
        last = (rc, (out.value.decode(errors="replace") if out.value else ""))
        if rc == 0 and out.value: return (f"{rc} (cb={cbv})", last[1])
    return last
import os
gens = [int(x, 0) for x in os.environ.get("GENS", "0x1000004").split(",")]
for g in gens:
    r = disasm(g)
    print("=== gen", hex(g), r[0] if isinstance(r, tuple) else r)
    if isinstance(r, tuple):
        open(f"isa_{kname}_{g:x}.txt", "w").write(r[1]); print(r[1][:1500])
