"""Time `matmul_sg`-style kernels directly through OpenCL (pyopencl), bypassing the plugin.

    python tools/matmul_bench.py kernels/kernels.cl "-DSOME_MACRO=1" 2048x2048x2048 4096x4096x4096   # MxNxK
    SIMD=8 TR=32 python tools/matmul_bench.py tools/matmul_variants.cl "-DPF=4 -DSWZ=8 -cl-intel-256-GRF-per-thread"

Reports the best of 30 timed runs (GPU event time) so transient contention does not mask the kernel, and
checks a sample of rows against float64. `tools/matmul_variants.cl` is the parameterized experiment kernel
(sub-group width, tile height, prefetch depth, tile order) the production kernel was chosen from.
Env: TR (tile rows, default 16), SIMD (sub-group width, default 16), LX/LY (sub-groups per work-group)."""
import os, statistics, sys
import numpy as np
import pyopencl as cl

src, opts = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
shapes = [tuple(int(v) for v in a.split("x")) for a in sys.argv[3:]] or [(2048,2048,2048),(4096,4096,4096),(256,2048,2048),(2048,256,2048)]
TR = int(os.environ.get("TR", 16)); SIMD = int(os.environ.get("SIMD", 16)); LX = int(os.environ.get("LX", 1)); LY = int(os.environ.get("LY", 1))
dev = [d for p in cl.get_platforms() for d in p.get_devices() if "Intel" in d.name][0]
ctx = cl.Context([dev]); q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
prog = cl.Program(ctx, open(src, encoding="utf-8").read()).build(options=f"-cl-intel-greater-than-4GB-buffer-required -DSG_TR={TR} -DSG_SIMD={SIMD} " + opts)
k = prog.matmul_sg
mf = cl.mem_flags
for M, N, K in shapes:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((M, K), dtype=np.float32); b = rng.standard_normal((K, N), dtype=np.float32)
    da = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a); db = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gx, gy = (N // 32) * SIMD, M // TR
    gws = (gx, gy); lws = (SIMD * LX, LY)
    assert gx % (SIMD * LX) == 0 and gy % LY == 0, "work-group does not divide the grid"
    def run():
        return k(q, gws, lws, da, db, dc, np.uint64(M), np.uint64(N), np.uint64(K))
    import time
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.0:
        run(); n += 1
        if n % 4 == 0: q.finish()
    q.finish()
    ts = []
    for _ in range(30):
        e = run(); e.wait(); ts.append((e.profile.end - e.profile.start) * 1e-9)
    t = min(ts)  # best of 30: transient contention only ever adds time
    c = np.empty((M, N), np.float32); cl.enqueue_copy(q, c, dc)
    rows = rng.integers(0, M, 8)
    ref = a[rows].astype(np.float64) @ b.astype(np.float64)
    err = float(np.abs(c[rows] - ref).max())
    print(f"{M:>5}x{K:>5}x{N:>5}  {2*M*K*N/t/1e9:7.0f} GFLOP/s  {t*1e3:7.2f} ms  err={err:.1e}", flush=True)
