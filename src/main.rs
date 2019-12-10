extern crate itertools;
extern crate minifb;
extern crate mnist;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use core::borrow::Borrow;
use std::iter::*;

use minifb::*;
use mnist::{Mnist, MnistBuilder};
use ndarray::s;
use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use libc;
use std::mem;
use std::collections::HashMap;
use std::f64;

const MAX_PERP_SEARCH_ITERS: u32 = 15;
const PERP: f64 = 30.0;
const LR: f64 = 50f64;
const MAX_SGD_ITERS: i32 = 1000;
const WIDTH: usize = 768;
const HEIGHT: usize = 768;
const TRN_SIZE: usize = 400;

type Data = Array2<f64>;
type DistanceMatrix = Array2<f64>; // (ndatum, ndatum)

fn distances(data: &Data) -> DistanceMatrix {
    let n: usize = data.shape()[0];
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            0.
        } else {
            let xi = data.slice(s![i, ..]);
            let xj = data.slice(s![j, ..]);
            norm_sq(&xi, &xj)
        }
    })
}

fn conditional_dist(dists: &DistanceMatrix, i: usize, beta: f64) -> Array1<f64> {
    let n = dists.shape()[0] as usize;
    let d = dists.slice(s![i, ..]);
    let all = Array1::from_shape_fn(n, |j| {
        if i == j {
            0.
        } else {
//            (-d[j] / (2. * sigma_i * sigma_i)).exp()
            (-d[j] * beta).exp()
        }
    });
    &all / all.sum()
}

fn joint_t_dist(dists: &DistanceMatrix) -> Array2<f64> {
    let n = dists.shape()[0] as usize;
    let all = Array2::from_shape_fn((n, n), |(k, l)| {
        if k == l {
            0.
        } else {
            1.0 / (1.0 + dists[[k, l]])
        }
    });
    &all / all.sum()
}

fn perp_search(dists: &DistanceMatrix, i: usize, max_iters: u32, target_perp: f64) ->
(Array1<f64>, f64) {
    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut beta = 1.0;

    let mut iter = 0;

    let mut dist = conditional_dist(&dists, i, beta);

    while iter < max_iters {
        let cand_perp = perp(&dist);

//        println!("{} {} {} -> {}", beta_min, beta, beta_max, cand_perp);
        if target_perp < cand_perp {
            beta_min = beta;
            if beta_max.is_infinite() {
                beta *= 2.0;
            } else {
                beta = (beta + beta_max) / 2.0;
            }
        } else {
            beta_max = beta;
            if beta_min.is_infinite() {
                beta /= 2.0;
            } else {
                beta = (beta + beta_min) / 2.0;
            }
        }

        iter += 1;

        dist = conditional_dist(&dists, i, beta);
    }

    (dist, beta)
//    let mut s_min = 0.000001f64;
//    let mut s_max = 1000f64;
//    let mut s = 1.0;
//    let mut iter = 0;
//    let mut dist = conditional_dist(&dists, i, s);
//    let mut prev_perp = 0.0;
//    while iter < max_iters {
////        println!("> {:?}", dist);
//        let cand_perp = perp(&dist);
//
//        if target_perp < cand_perp {
//            s_max = s;
//        } else {
//            s_min = s;
//        }
//        s = s_min + (s_max - s_min) / 2.;
////        s = (s_max + s_min) / 2.0;
//
//        iter += 1;
//
//        dist = conditional_dist(&dists, i, s);
//
//        if (cand_perp - prev_perp).abs() < 0.0000001 {
//            break;
//        }
//        prev_perp = cand_perp;
//    }
//
//    (dist, s)
}

fn symmetrised_dist_search(dists: &DistanceMatrix, target_perp: f64) -> Array2<f64> {
    let n = dists.shape()[0] as usize;

    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        let (p_ji, _s_i) = perp_search(&dists, i, MAX_PERP_SEARCH_ITERS, target_perp);
        result.row_mut(i).assign(&p_ji);
    }

    (&result + &result.t()) / (2.0f64 * (n as f64))
}

fn entropy(dist: &Array1<f64>) -> f64 {
    let ref log_dist = dist.mapv(|x: f64|
        if x < 1e-8 {
            0.
        } else {
            x.log2()
        }
    );
    -(dist * log_dist).sum()
}

fn perp(dist: &Array1<f64>) -> f64 {
    2.0_f64.powf(entropy(&dist))
}

fn grad(proj: &Array2<f64>, p_ij: &Array2<f64>, q_ij: &Array2<f64>,
        lo_dists: &Array2<f64>)
        -> Array2<f64> {
    let n = proj.shape()[0] as usize;
    // result is same size as proj
    let mut result = Array::zeros(proj.raw_dim());
    let p_minus_q = p_ij.borrow() - q_ij.borrow();

    for i in 0..n {
        let mut sum = Array1::zeros(2);
        for j in 0..n {
            if i == j {
                continue;
            }

            sum = sum + (p_minus_q[[i, j]])
                * (proj.row(i).borrow() - proj.row(j).borrow())
                / (1. + lo_dists[[i, j]]);
        }
        sum *= 4.0;
        result.row_mut(i).assign(&sum);
    }
    result + &(proj * 0.0001)
}

fn norm_sq(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    (a - b).dot(&(a - b))
}

fn vmin(v: &ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0f64 / 0.0, |x, y| {
        f64::min(x, *y)
    })
}

fn vmax(v: &ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0f64 / 0.0, |x, y| {
        f64::max(x, *y)
    })
}

fn update_proj(buf: &mut Vec<u32>, proj: &Array2<f64>, lbls: &Vec<u8>) {
    let mut lbl_map = HashMap::new();
    lbl_map.insert(0, 0xff_33_ee_22);
    lbl_map.insert(1, 0xff_83_33_fc);
    lbl_map.insert(2, 0xff_ad_00_bb);
    lbl_map.insert(3, 0xff_cc_dd_30);
    lbl_map.insert(4, 0xff_8a_c3_30);
    lbl_map.insert(5, 0xff_99_a0_b4);
    lbl_map.insert(6, 0xff_da_b3_03);
    lbl_map.insert(7, 0xff_03_dd_a3);
    lbl_map.insert(8, 0xff_44_00_ff);
    lbl_map.insert(9, 0xff_ff_8a_a8);
    unsafe {
        libc::memset(buf.as_mut_ptr() as _, 0, buf.len() * mem::size_of::<u32>());
    }
    let xs = proj.slice(s![.., 0]);
    let ys = proj.slice(s![.., 1]);
    let min_x = vmin(&xs);
    let max_x = vmax(&xs);
    let min_y = vmin(&ys);
    let max_y = vmax(&ys);
//    println!("xs {}", xs);
//    println!("min x {} max x {}", min_x, max_x);
    for (i, pt) in proj.outer_iter().enumerate() {
//        println!("pt: {}", pt);
        let px = (pt[0] - min_x) / (max_x - min_x);
        let py = (pt[1] - min_y) / (max_y - min_y);

        let xx = (px * WIDTH as f64) as usize;
        let yy = (py * HEIGHT as f64) as usize;

        let lbl_colour = *lbl_map.get(&(lbls[i] as u32)).unwrap();
        if xx > 0 && yy > 0 && xx < WIDTH - 1 && yy < HEIGHT - 1 {
            buf[yy * WIDTH + xx - 1] = lbl_colour;
            buf[(yy - 1) * WIDTH + xx] = lbl_colour;
            buf[(yy + 1) * WIDTH + xx] = lbl_colour;
            buf[yy * WIDTH + xx + 1] = lbl_colour;
            buf[yy * WIDTH + xx] = lbl_colour;
        }
    }
}

fn main() {
    let (trn_size, rows, cols) = (TRN_SIZE, 28, 28);

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize();

    let u8_images = Array::from_vec(trn_img)
        .into_shape((trn_size as usize, (rows * cols) as usize)).unwrap();
    let images = u8_images.mapv(|f| {
        (f as f64) / 255.
    });

    let mut buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = match Window::new("Test", WIDTH, HEIGHT, WindowOptions::default()) {
        Ok(win) => win,
        Err(err) => {
            println!("Unable to create window {}", err);
            return;
        }
    };

    let normal = Normal::new(0., 1e-4);
    let mut proj = Array::random((trn_size as usize, 2), normal);
    let mut velocity_prev: Array2<f64> = Array2::zeros((trn_size as usize, 2));
    let dists = distances(&images);
    let mut p_ij = symmetrised_dist_search(&dists, PERP) * 12.0;

    for i in 0..MAX_SGD_ITERS {
        let lo_dists = distances(&proj);
        let q_ij = joint_t_dist(&lo_dists);

        println!("{}", i);
        if i == 150 {
            p_ij /= 12.0;
        }

        let del: Array2<f64> = grad(&proj, &p_ij, &q_ij, &lo_dists);
        let velocity = LR * &del + 0.9 * &velocity_prev;
        velocity_prev.assign(&velocity);
        proj = &proj - &velocity;

        let avg: Array1<f64> = proj.sum_axis(Axis(0)) / proj.len() as f64;
        proj = &proj - &avg;

        update_proj(&mut buf, &proj, &trn_lbl);
        window.update_with_buffer_size(&buf, WIDTH, HEIGHT).unwrap();
    }
}
