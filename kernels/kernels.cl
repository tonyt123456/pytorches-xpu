// OpenCL C kernels for the PyTorches xpu (Intel GPU) plugin.
// All indexing is 64-bit (ulong) so tensors / buffers larger than 4 GiB work. The program is
// built with -cl-intel-greater-than-4GB-buffer-required (stateless 64-bit addressing).
// No native_* / fast-math variants are used: results track the CPU reference closely.

typedef struct {
    uint ndim;
    uint pad;
    ulong shape[8];
    ulong stride[8];
} Dims;

// Offset of the element for linear (row-major) output index i, with 64-bit math.
static inline ulong off64(ulong i, __private const Dims* d) {
    ulong off = 0;
    for (int k = (int)d->ndim - 1; k >= 0; k--) {
        ulong s = d->shape[k];
        ulong q = i / s;
        off += (i - q * s) * d->stride[k];
        i = q;
    }
    return off;
}

// Same with 32-bit math (64-bit division is emulated and slow); the host guarantees that
// the element count and every reachable offset fit in 32 bits when it selects this path.
static inline ulong off32(ulong i64, __private const Dims* d) {
    uint i = (uint)i64;
    uint off = 0;
    for (int k = (int)d->ndim - 1; k >= 0; k--) {
        uint s = (uint)d->shape[k];
        uint q = i / s;
        off += (i - q * s) * (uint)d->stride[k];
        i = q;
    }
    return (ulong)off;
}

static inline ulong offs(ulong i, __private const Dims* d, int use32) {
    return use32 ? off32(i, d) : off64(i, d);
}

// op codes are the ABI op codes mapped on the host: 0 = copy, 1 NEG, 2 EXP, 3 LOG, 4 RELU, 5 TANH, 6 STEP
static inline float un(int op, float v) {
    switch (op) {
        case 1: return -v;
        case 2: return exp(v);
        case 3: return log(v);
        case 4: return fmax(v, 0.0f);
        case 5: return tanh(v);
        case 6: return v > 0.0f ? 1.0f : 0.0f;
        default: return v;
    }
}

// 0 ADD 1 SUB 2 MUL 3 DIV
static inline float bi(int op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        default: return a / b;
    }
}

__kernel void unary_flat(__global const float* x, __global float* y, ulong n, int op) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) y[i] = un(op, x[i]);
}

__kernel void unary_strided(__global const float* x, __global float* y, ulong n, int op, Dims d, int use32) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) y[i] = un(op, x[offs(i, &d, use32)]);
}

__kernel void binary_flat(__global const float* a, __global const float* b, __global float* y, ulong n, int op) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) y[i] = bi(op, a[i], b[i]);
}

__kernel void binary_strided(__global const float* a, __global const float* b, __global float* y, ulong n,
                             int op, Dims da, Dims db, int use32) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0))
        y[i] = bi(op, a[offs(i, &da, use32)], b[offs(i, &db, use32)]);
}

// Tiled 2-D transpose through local memory: `in` is a contiguous [R, C] matrix, `out` becomes the
// contiguous [C, R] transpose. 32x32 tiles, 32x8 work-group; the +1 pad avoids bank conflicts, and both
// the read and the write are coalesced.
#define TP 32
__kernel __attribute__((reqd_work_group_size(32, 8, 1)))
void transpose2d(__global const float* in, __global float* out, ulong R, ulong C) {
    __local float tile[TP][TP + 1];
    const uint lx = get_local_id(0), ly = get_local_id(1);
    const ulong cx = (ulong)get_group_id(0) * TP + lx;   // source column
    const ulong ry = (ulong)get_group_id(1) * TP + ly;   // source row (first of 4)
    for (uint j = 0; j < TP; j += 8)
        if (cx < C && ry + j < R) tile[ly + j][lx] = in[(ry + j) * C + cx];
    barrier(CLK_LOCAL_MEM_FENCE);
    const ulong ox = (ulong)get_group_id(1) * TP + lx;   // output column = source row
    const ulong oy = (ulong)get_group_id(0) * TP + ly;   // output row = source column (first of 4)
    for (uint j = 0; j < TP; j += 8)
        if (ox < R && oy + j < C) out[(oy + j) * R + ox] = tile[lx][ly + j];
}

// y = a + alpha * b (fused update; y may alias a, so no restrict).
__kernel void axpy(__global const float* a, __global const float* b, __global float* y, ulong n, float alpha) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) y[i] = a[i] + alpha * b[i];
}

__kernel void fill(__global float* y, ulong n, float v) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) y[i] = v;
}

static inline ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}

__kernel void rand_normal(__global float* y, ulong n, ulong seed) {
    for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) {
        ulong h = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15UL));
        float u1 = (float)((h >> 40) + 1) / 16777216.0f;
        float u2 = (float)(h & 0xFFFFFFUL) / 16777216.0f;
        y[i] = sqrt(-2.0f * log(u1)) * cos(6.2831855f * u2);
    }
}

// ---- matmul: C[M,N] = A[M,K] * B[K,N], row-major -------------------------------------------
// 64x64 output tile per work-group of 16x16 threads, 4x4 register micro-tile per thread,
// BK-deep K slices staged through local memory.
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void matmul(__global const float* A, __global const float* B, __global float* C, ulong M, ulong N, ulong K,
            ulong roff) {
    // `roff`: first output row to compute (A and C are advanced to it; M is then the number of rows).
    A += roff * K;
    C += roff * N;
    __local float As[BK][BM + 1];  // As[k][m]
    __local float Bs[BK][BN];      // Bs[k][n]
    const uint tx = get_local_id(0), ty = get_local_id(1);
    const uint tid = ty * 16 + tx;
    const ulong row0 = (ulong)get_group_id(1) * BM;
    const ulong col0 = (ulong)get_group_id(0) * BN;

    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (ulong k0 = 0; k0 < K; k0 += BK) {
        for (int l = 0; l < 4; l++) {
            uint idx = tid + l * 256;
            uint r = idx / BK, c = idx % BK;
            ulong gr = row0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int l = 0; l < 4; l++) {
            uint idx = tid + l * 256;
            uint r = idx / BN, c = idx % BN;
            ulong gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[kk][ty + i * 16];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx + j * 16];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = mad(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; i++) {
        ulong gr = row0 + ty + i * 16;
        if (gr >= M) continue;
        for (int j = 0; j < TN; j++) {
            ulong gc = col0 + tx + j * 16;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// ---- matmul fast path: sub-group tiles, no local memory ---------------------------------------
// Needs N % 32 == 0, K % 16 == 0, a whole number of 16-row blocks, M*K, K*N and M*N below 2^32 elements
// (the host checks; a leftover of fewer than 16 rows goes to the general kernel above) and
// cl_intel_subgroups. A 16-lane sub-group computes a 16 x 32 tile of C: lane j owns columns j and j+16 of
// every tile row. B rows are fetched with one wide block read (lane j gets B[k][j] and B[k][j+16]);
// A[r][k0+lane] is block-read and each element is broadcast to all lanes by sub_group_broadcast.
//
// What it took to go from 2.4 to ~3.5 TFLOP/s on an Arc 140T (fp32 compute ceiling of this kernel's
// FMA stream, with the loads removed, measured at 4.7-4.9; PyTorch reaches ~4.3). Found by reading the
// compiled ISA (IGA disassembly of the program binary), not by guessing:
//  * B loads were issued right before their FMAs, so every k step stalled on a full cache round trip.
//    They now run SG_PF steps ahead through a small register ring. Deeper rings spill the 128-register
//    file (acc 64 + A 32 + ring), which costs far more than it saves.
//  * The order work-groups run in decides what the cache sees. Plain raster order streams all of B for
//    every wave of tiles; grouped order (SG_GROUP tile-rows at a time) keeps concurrent tiles on a
//    compact block of A and B. This alone is worth 1.5-1.7x at 4096.
//  * 32-bit element offsets: 64-bit index math compiled to ~7 scalar instructions per A load and ~30 per
//    B load. (On its own this changed nothing; the loads above were the limit.)
// Remaining gap to PyTorch: the kernel is now bound by load bandwidth (192 bytes per 1024 FLOP). A 32-row
// tile halves that but needs the 256-register mode, where this compiler spills; see benchmarks/README.md.
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define SG_TR 16
#define SG_TC 2
#define SG_PF 2
#define SG_GROUP 8
__kernel __attribute__((intel_reqd_sub_group_size(16)))
void matmul_sg(__global const float* A, __global const float* B, __global float* C, ulong M, ulong N, ulong K) {
    const uint lane = get_sub_group_local_id();
    const uint N32 = (uint)N, K32 = (uint)K;
    // Grouped tile order. One work-group is one sub-group (the host launches 16x1), so the group id is
    // the tile id: `lin` is the order the hardware dispatches tiles in.
    const uint tiles_x = get_num_groups(0), tiles_y = get_num_groups(1);
    const uint lin = get_group_id(1) * tiles_x + get_group_id(0);
    const uint per_group = SG_GROUP * tiles_x;
    const uint first_row_tile = (lin / per_group) * SG_GROUP;
    const uint rows_in_group = min(tiles_y - first_row_tile, (uint)SG_GROUP);
    const uint col0 = ((lin % per_group) / rows_in_group) * (16 * SG_TC);
    const uint row0 = (first_row_tile + (lin % per_group) % rows_in_group) * SG_TR;

    float acc[SG_TR][SG_TC];
    for (int r = 0; r < SG_TR; r++)
        for (int c = 0; c < SG_TC; c++) acc[r][c] = 0.0f;
    const __global uint* Au = (const __global uint*)A;
    const __global uint* Bu = (const __global uint*)B;
    const uint aoff = row0 * K32;
    uint boff = col0;
    for (uint k0 = 0; k0 < K32; k0 += 16) {
        float a[SG_TR];
#pragma unroll
        for (int r = 0; r < SG_TR; r++) a[r] = as_float(intel_sub_group_block_read(Au + (aoff + r * K32 + k0)));
        uint2 q[SG_PF];
#pragma unroll
        for (int d = 0; d < SG_PF; d++) q[d] = intel_sub_group_block_read2(Bu + (boff + d * N32));
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const uint2 u = q[kk % SG_PF];
            if (kk + SG_PF < 16) q[kk % SG_PF] = intel_sub_group_block_read2(Bu + (boff + (kk + SG_PF) * N32));
            const float b0 = as_float(u.s0), b1 = as_float(u.s1);
#pragma unroll
            for (int r = 0; r < SG_TR; r++) {
                const float av = sub_group_broadcast(a[r], kk);
                acc[r][0] = mad(av, b0, acc[r][0]);
                acc[r][1] = mad(av, b1, acc[r][1]);
            }
        }
        boff += 16 * N32;
    }
    for (int r = 0; r < SG_TR; r++)
        for (int c = 0; c < SG_TC; c++) C[(row0 + r) * N32 + col0 + c * 16 + lane] = acc[r][c];
}
#endif

// ---- sum over one axis of an (outer, n, inner) contiguous tensor -----------------------------
// Partial results use the layout out[(o*inner+i)*splits + s]; when splits == 1 that is the
// final output. reduce_partials folds the splits.

// inner == 1: one work-group per (row, split), 256-thread tree reduction.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sum_rows(__global const float* x, __global float* out, ulong outer, ulong n, ulong splits, ulong chunk) {
    __local float sh[256];
    const uint lid = get_local_id(0);
    const ulong total = outer * splits;
    for (ulong g = get_group_id(0); g < total; g += get_num_groups(0)) {
        ulong o = g / splits, s = g - o * splits;
        ulong lo = s * chunk;
        ulong hi = min(n, lo + chunk);
        __global const float* p = x + o * n;
        float acc = 0.0f;
        // 4-wide loads (vload4 needs only element alignment), then a scalar tail.
        const ulong cnt = hi - lo, n4 = cnt / 4;
        for (ulong j = lid; j < n4; j += 256) {
            const float4 v = vload4(j, p + lo);
            acc += (v.x + v.y) + (v.z + v.w);
        }
        for (ulong t = n4 * 4 + lid; t < cnt; t += 256) acc += p[lo + t];
        sh[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 128; st > 0; st >>= 1) {
            if (lid < st) sh[lid] += sh[lid + st];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[g] = sh[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// general: one thread per (split, outer, inner); neighbouring threads read neighbouring inner elements.
__kernel void sum_inner(__global const float* x, __global float* out, ulong outer, ulong n, ulong inner,
                        ulong splits, ulong chunk) {
    const ulong total = outer * inner * splits;
    for (ulong t = get_global_id(0); t < total; t += get_global_size(0)) {
        ulong i = t % inner;
        ulong r = t / inner;
        ulong o = r % outer;
        ulong s = r / outer;
        ulong lo = s * chunk;
        ulong hi = min(n, lo + chunk);
        float acc = 0.0f;
        for (ulong j = lo; j < hi; j++) acc += x[(o * n + j) * inner + i];
        out[(o * inner + i) * splits + s] = acc;
    }
}

__kernel void reduce_partials(__global const float* part, __global float* out, ulong count, ulong splits) {
    for (ulong t = get_global_id(0); t < count; t += get_global_size(0)) {
        float acc = 0.0f;
        for (ulong s = 0; s < splits; s++) acc += part[t * splits + s];
        out[t] = acc;
    }
}
