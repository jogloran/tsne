extern crate itertools;
extern crate minifb;
extern crate mnist;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use core::borrow::Borrow;
use std::iter::*;

use itertools::Itertools;
use minifb::*;
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Shape};
use ndarray::Array;
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, Range};
use libc;
use std::mem;
use std::collections::HashMap;

const PERP: u32 = 50;

type Data = Array2<f64>;
type DistanceMatrix = Array2<f64>; // (ndatum, ndatum)

fn distances(data: &Data) -> DistanceMatrix {
    let n: usize = data.shape()[0];
    Array2::from_shape_fn((n, n), |(i, j)| {
        let xi = data.slice(s![i, ..]);
        let xj = data.slice(s![j, ..]);
        norm_sq(&xi, &xj)
    })
}

fn conditional_dist(dists: &DistanceMatrix, i: usize, sigma_i: f64) -> Array1<f64> {
    let n = dists.shape()[0] as usize;
    let d = dists.slice(s![i, ..]);
    let mut all = Array1::from_shape_fn(n, |j| {
        (-d[j] / (2. * sigma_i * sigma_i)).exp()
    });
    all[i] = 0.0;
    all /= all.sum();
    all
}

fn joint_t_dist(dists: &DistanceMatrix) -> Array2<f64> {
    let n = dists.shape()[0] as usize;
    let mut all = Array2::from_shape_fn((n, n), |(k, l)| {
        1.0 / (1.0 + dists[[k, l]])
    });
    all.diag_mut().fill(0.);
    let denom = all.sum() - all.diag().sum();
    all / denom
}

fn perp_search(dists: &DistanceMatrix, data: &Data, i: usize, max_iters: u32, target_perp: f64) ->
(Array1<f64>, f64) {
    let mut s_min = 0.001;
    let mut s_max = 100.;
    let mut s = s_min + (s_max - s_min) / 2.;
    let mut iter = 0;
    let mut dist = conditional_dist(&dists, i, s);
    while iter < max_iters {
//        println!("> {:?}", dist);
        let cand_perp = perp(&dist);
        if target_perp < cand_perp {
            s_max = s;
        } else {
            s_min = s;
        }
        s = s_min + (s_max - s_min) / 2.;
        iter += 1;
//        println!("iter: {} s: {}, perp: {}", iter, s, cand_perp);

        dist = conditional_dist(&dists, i, s);
    }

    (dist, s)
}

fn conditional_dist_search(dists: &DistanceMatrix, data: &Data, target_perp: f64) -> Array2<f64> {
    let n = dists.shape()[0] as usize;

    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        let (p_ji, s_i) = perp_search(&dists, &data, i, PERP, target_perp);
        result.row_mut(i).assign(&p_ji);
    }

    result
}

fn symmetrised_dist_search(dists: &DistanceMatrix, data: &Data, target_perp: f64) -> Array2<f64> {
    // for each datum x_i,
    // i is fixed
    let n = dists.shape()[0] as usize;

    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        let (p_ji, s_i) = perp_search(&dists, &data, i, PERP, target_perp);
        result.row_mut(i).assign(&p_ji);
    }
    let mut result2 = Array2::zeros((n, n));
    for j in 0..n {
        let (p_ji, s_i) = perp_search(&dists, &data, j, PERP, target_perp);
        result2.row_mut(j).assign(&p_ji);
    }

    let mut symmetrised = Array2::zeros((n, n));
    symmetrised = (result + result2.t()) / 2.0;
    symmetrised
}

fn H(dist: &Array1<f64>) -> f64 {
    let ref log_dist = dist.mapv(|x: f64| (x + 1e-8).log2());
    -(dist * log_dist).sum()
}

fn perp(dist: &Array1<f64>) -> f64 {
    2.0_f64.powf(H(&dist))
}

fn grad(proj: &Array2<f64>, p_ij: &Array2<f64>, q_ij: &Array2<f64>)
        -> Array2<f64> {
    let n = proj.shape()[0] as usize;
    // result is same size as proj
    let mut result = Array::zeros(proj.raw_dim());

    for i in 0..n {
        let mut sum = Array1::zeros(2);
        for j in 0..n {
            if i == j {
                break;
            }
//            println!("1 {}", &proj.row(i));
//            println!("2 {}", &proj.row(j));
//            println!("a {:?}", &proj.row(i) - &proj.row(j));
//            println!("b {}", norm_sq(&proj.row(i), &proj.row(j)));
//            println!("c {}", 4. * (p_ij[[i, j]] - q_ij[[i, j]])
//                * (proj.row(i).borrow() - proj.row(j).borrow())
//                / (1. + norm_sq(&proj.row(i), &proj.row(j))));
            sum = sum + 4. * (p_ij[[i, j]] - q_ij[[i, j]])
                * (proj.row(i).borrow() - proj.row(j).borrow())
                / (1. + norm_sq(&proj.row(i), &proj.row(j)));
        }
//        println!("sum {}", sum);

        result.row_mut(i).assign(&sum);
    }

    result
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
        libc::memset(
            buf.as_mut_ptr() as _,
            0,
            buf.len() * mem::size_of::<u32>(),
        );
    }
    let xs = proj.slice(s![.., 0]);
    let ys = proj.slice(s![.., 1]);
    let min_x = vmin(&xs);
    let max_x = vmax(&xs);
    let min_y = vmin(&ys);
    let max_y = vmax(&ys);
//    println!("xs {}", xs);
//    println!("min x {} max {}", min_x, max_x);
    for (i, pt) in proj.outer_iter().enumerate() {
//        println!("pt: {}", pt);
        let px = (pt[0] - min_x) / (max_x - min_x);
        let py = (pt[1] - min_y) / (max_y - min_y);
        assert!(px >= 0. && px <= 1.);
        assert!(py >= 0. && py <= 1.);
//        println!("px {} py {}", px, py);
        let xx = (px * 639.) as usize;
        let yy = (py * 399.) as usize;

        if (xx > 0 && yy > 0 && xx < 639 && yy < 399) {
            buf[yy * 639 + xx - 1] = *lbl_map.get(&(lbls[i] as u32)).unwrap();
            buf[(yy - 1) * 639 + xx] = *lbl_map.get(&(lbls[i] as u32)).unwrap();
            buf[(yy + 1) * 639 + xx] = *lbl_map.get(&(lbls[i] as u32)).unwrap();
            buf[yy * 639 + xx + 1] = *lbl_map.get(&(lbls[i] as u32)).unwrap();
        }
        buf[yy * 639 + xx] = *lbl_map.get(&(lbls[i] as u32)).unwrap();
    }
}

fn main() {
    let (trn_size, rows, cols) = (150, 28, 28);
    let size: i32 = trn_size * rows * cols;

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize();
    println!("lbl {:?}", trn_lbl);

    let mut u8_images = Array::from_vec(trn_img)
        .into_shape((trn_size as usize, (rows * cols) as usize)).unwrap();
    let mut images = u8_images.mapv(|f| {
        (f as f64) / 255.
    });

    //    println!("img {:?}", images.slice(s![1, ..]));
    const lr: f64 = 10f64;
    const MAX_IT: i32 = 1000;

    let mut buf: Vec<u32> = vec![0; 640 * 400];
    let mut window = match Window::new("Test", 640, 400, WindowOptions::default()) {
        Ok(win) => win,
        Err(err) => {
            println!("Unable to create window {}", err);
            return;
        }
    };

    let mut it = 0;
    let normal = Normal::new(0., 1e-4);
    let mut proj = Array::random((trn_size as usize, 2), normal);
    for i in 0..MAX_IT {
        let mut dists = distances(&images);
        let mut p_ij = symmetrised_dist_search(&dists, &images, 2.0);
        let mut lo_dists = distances(&proj);
        let mut q_ij = joint_t_dist(&lo_dists);
//    let mut proj = Array2::zeros((trn_size as usize, 2));
//        println!("proj: {}", proj);

        let del: Array2<f64> = grad(&proj, &p_ij, &q_ij);
//        println!("p_ij: {}", p_ij);
//        println!("q_ij: {}", q_ij);
//        println!("{} del: {}", i, del);
        proj = proj + lr * del;
//        println!("{} proj: {}", i, proj);

        update_proj(&mut buf, &proj, &trn_lbl);
        window.update_with_buffer_size(&buf, 640, 400).unwrap();
    }
}
