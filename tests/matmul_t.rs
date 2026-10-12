//! `MATMUL_T` through the plugin's vtable, for every transpose-flag combination and for shapes that take
//! each internal route (plain, transposed operand, swapped `C^T = B A^T`, leftover rows, shapes the fast
//! kernel does not accept). Compared with a straightforward host matmul. Skips without an Intel GPU.

use libloading::Library;
use pytorches_plugin_abi::*;
use std::ffi::c_void;
use std::ptr::null_mut;

mod common;

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 16777216.0) * 2.0 - 1.0
        })
        .collect()
}

/// `C[m,n] = op(A) op(B)` computed on the host from stored row-major operands.
fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, a_t: bool, b_t: bool) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                let x = if a_t { a[p * m + i] } else { a[i * k + p] };
                let y = if b_t { b[j * k + p] } else { b[p * n + j] };
                acc += x as f64 * y as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

#[test]
fn matmul_t_matches_host_reference() {
    unsafe {
        let Ok(lib) = Library::new(common::xpu_dll()) else {
            eprintln!("SKIP: xpu dll not built");
            return;
        };
        let entry: unsafe extern "C" fn() -> *const PluginVTable = *lib.get(ENTRY_SYMBOL).unwrap();
        let vt = &*entry();
        if (vt.device_count)() == 0 {
            eprintln!("SKIP: no Intel OpenCL GPU");
            return;
        }
        assert_ne!(
            (vt.supports_op)(op::MATMUL_T),
            0,
            "plugin must advertise MATMUL_T"
        );

        let upload = |data: &[f32]| {
            let mut p: *mut c_void = null_mut();
            assert_eq!((vt.alloc)(0, data.len() * 4, &mut p), STATUS_OK);
            assert_eq!(
                (vt.copy_from_host)(0, p, data.as_ptr() as *const c_void, data.len() * 4),
                STATUS_OK
            );
            p
        };
        let desc = |p: *mut c_void, shape: &[u64], strides: &[u64]| TensorDesc {
            data: p,
            dtype: DTYPE_F32,
            ndim: 2,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
        };

        // (m, k, n): m is the rows of C, k the contracted size, n the columns.
        let shapes = [
            (256, 512, 512), // small batch against large weights: the swapped route for A B^T
            (64, 1024, 512), // same, fast kernel applies to the swapped product
            (512, 64, 512),  // large batch: B^T is the cheaper one to move
            (100, 128, 96),  // leftover rows (100 % 16 != 0)
            (17, 48, 33),    // nothing for the fast kernel
            (16, 16, 32),    // smallest fast tile
            (48, 32, 64),
            (33, 64, 32), // fast kernel needs n % 32 == 0 for the product and the swapped one
            (1, 1, 1),
            (3, 1000, 130),
            (256, 2048, 2048), // the MLP backward shape
        ];
        for (m, k, n) in shapes {
            for (a_t, b_t) in [(false, false), (true, false), (false, true), (true, true)] {
                let a_shape: [u64; 2] = if a_t {
                    [k as u64, m as u64]
                } else {
                    [m as u64, k as u64]
                };
                let b_shape: [u64; 2] = if b_t {
                    [n as u64, k as u64]
                } else {
                    [k as u64, n as u64]
                };
                let a = lcg(m * k, 11);
                let b = lcg(k * n, 12);
                let (da, db) = (upload(&a), upload(&b));
                let mut dc: *mut c_void = null_mut();
                assert_eq!((vt.alloc)(0, m * n * 4, &mut dc), STATUS_OK);

                let c_shape = [m as u64, n as u64];
                let a_strides = [a_shape[1], 1];
                let b_strides = [b_shape[1], 1];
                let c_strides = [n as u64, 1];
                let ins = [
                    desc(da, &a_shape, &a_strides),
                    desc(db, &b_shape, &b_strides),
                ];
                let outs = [desc(dc, &c_shape, &c_strides)];
                let attrs = OpAttrs {
                    ints: [(a_t as i64) | ((b_t as i64) << 1), 0, 0, 0],
                };
                let st = (vt.execute)(0, op::MATMUL_T, &attrs, ins.as_ptr(), 2, outs.as_ptr(), 1);
                assert_eq!(st, STATUS_OK, "execute {m}x{k}x{n} a_t={a_t} b_t={b_t}");
                assert_eq!((vt.synchronize)(0), STATUS_OK);

                let mut got = vec![0.0f32; m * n];
                assert_eq!(
                    (vt.copy_to_host)(0, got.as_mut_ptr() as *mut c_void, dc, m * n * 4),
                    STATUS_OK
                );
                let want = reference(&a, &b, m, k, n, a_t, b_t);
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (x - y).abs() <= 2e-4 * (1.0 + y.abs()),
                        "{m}x{k}x{n} a_t={a_t} b_t={b_t}: element {i}: got {x}, want {y}"
                    );
                }
                (vt.free)(0, da);
                (vt.free)(0, db);
                (vt.free)(0, dc);
            }
        }
    }
}
