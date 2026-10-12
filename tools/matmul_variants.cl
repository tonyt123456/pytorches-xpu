// Experimental sub-group matmul with a selectable sub-group width. Build with -DSG_SIMD=8|16 -DSG_TR=.. -DPF=.. -DSWZ=..
// Tile per sub-group: SG_TR rows x 32 cols. Lane j owns columns j, j+SG_SIMD, ... (SG_TC = 32/SG_SIMD of them).
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#ifndef SG_SIMD
#define SG_SIMD 16
#endif
#ifndef SG_TR
#define SG_TR 16
#endif
#ifndef PF
#define PF 2
#endif
#ifndef SWZ
#define SWZ 8
#endif
#define SG_TC (32 / SG_SIMD)

#if SG_TC == 2
#define BREAD(p) intel_sub_group_block_read2(p)
typedef uint2 bvec;
#elif SG_TC == 4
#define BREAD(p) intel_sub_group_block_read4(p)
typedef uint4 bvec;
#endif

static inline float bcomp(bvec u, int c) {
#if SG_TC == 2
    return as_float(c == 0 ? u.s0 : u.s1);
#else
    return as_float(c == 0 ? u.s0 : c == 1 ? u.s1 : c == 2 ? u.s2 : u.s3);
#endif
}

__kernel __attribute__((intel_reqd_sub_group_size(SG_SIMD)))
void matmul_sg(__global const float* A, __global const float* B, __global float* C, ulong M, ulong N, ulong K) {
    const uint lane = get_sub_group_local_id();
    const uint N32 = (uint)N, K32 = (uint)K;
    const uint tiles_x = get_num_groups(0), tiles_y = get_num_groups(1);
    const uint lin = get_group_id(1) * tiles_x + get_group_id(0);
    const uint per_group = SWZ * tiles_x;
    const uint first_row_tile = (lin / per_group) * SWZ;
    const uint rows_in_group = min(tiles_y - first_row_tile, (uint)SWZ);
    const uint col0 = ((lin % per_group) / rows_in_group) * 32;
    const uint row0 = (first_row_tile + (lin % per_group) % rows_in_group) * SG_TR;

    float acc[SG_TR][SG_TC];
    for (int r = 0; r < SG_TR; r++)
        for (int c = 0; c < SG_TC; c++) acc[r][c] = 0.0f;
    const __global uint* Au = (const __global uint*)A;
    const __global uint* Bu = (const __global uint*)B;
    const uint aoff = row0 * K32;
    uint boff = col0;
    for (uint k0 = 0; k0 < K32; k0 += SG_SIMD) {
        float a[SG_TR];
#pragma unroll
        for (int r = 0; r < SG_TR; r++) a[r] = as_float(intel_sub_group_block_read(Au + (aoff + r * K32 + k0)));
        bvec q[PF];
#pragma unroll
        for (int d = 0; d < PF; d++) q[d] = BREAD(Bu + (boff + d * N32));
#pragma unroll
        for (int kk = 0; kk < SG_SIMD; kk++) {
            const bvec u = q[kk % PF];
            if (kk + PF < SG_SIMD) q[kk % PF] = BREAD(Bu + (boff + (kk + PF) * N32));
#pragma unroll
            for (int r = 0; r < SG_TR; r++) {
                const float av = sub_group_broadcast(a[r], kk);
#pragma unroll
                for (int c = 0; c < SG_TC; c++) acc[r][c] = mad(av, bcomp(u, c), acc[r][c]);
            }
        }
        boff += SG_SIMD * N32;
    }
    for (int r = 0; r < SG_TR; r++)
        for (int c = 0; c < SG_TC; c++) C[(row0 + r) * N32 + col0 + c * SG_SIMD + lane] = acc[r][c];
}
