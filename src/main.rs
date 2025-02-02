extern crate itertools;
extern crate minifb;
extern crate mnist;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_parallel;
extern crate rand;
extern crate rayon;

use core::borrow::Borrow;
use std::iter::*;

use rand::Rng;
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
use rayon::prelude::*;
//use ndarray::parallel::prelude::*;
use ndarray_parallel::NdarrayIntoParallelIterator;
use std::time::{Duration, Instant};
use ndarray_parallel::prelude::ParMap;

const MAX_PERP_SEARCH_ITERS: u32 = 25;
const PERP: f64 = 15.0;
const LR: f64 = 15f64;
const MAX_SGD_ITERS: usize = 10000;
const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;
const TRN_SIZE: usize = 2500;
const SIMILARITY_EXAGG_FACTOR: f64 = 12.0;
const SIMILARITY_EXAGG_STOP_ITER: usize = 100;
const IMG_WIDTH: usize = 28;
const IMG_HEIGHT: usize = 28;
const SHOW_IMGS: bool = false;

const MAP_COLOURS: [u32; 10] = [
    0xff_33_ee_22,
    0xff_83_33_fc,
    0xff_ad_00_bb,
    0xff_cc_dd_30,
    0xff_8a_c3_30,
    0xff_99_a0_b4,
    0xff_da_b3_03,
    0xff_03_dd_a3,
    0xff_44_00_ff,
    0xff_ff_8a_a8
];

type Data = Array2<f64>;
type Joint = Array2<f64>;
type DistanceMatrix = Array2<f64>; // (ndatum, ndatum)

fn distances(data: &Data) -> DistanceMatrix {
    let n: usize = data.shape()[0];

    let mut vs = (0..n * n).into_par_iter().map(|idx| {
        let i = idx / n;
        let j = idx % n;
        if i == j {
            0.
        } else {
            let xi = data.slice(s![i, ..]);
            let xj = data.slice(s![j, ..]);
            norm_sq(&xi, &xj)
        }
    }).collect();

    Array2::from_shape_vec((n, n), vs).unwrap()
}

fn distances_serial(data: &Data) -> DistanceMatrix {
    let n = data.shape()[0] as usize;
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
            (-d[j] * beta).exp()
        }
    });
    &all / all.sum()
}

fn joint_t_dist(dists: &DistanceMatrix) -> Joint {
//    let mut result = dists.clone();
//    result.par_mapv_inplace(|v| {
//        1.0 / (1.0 + v)
//    });
//    result /= result.sum();
//    result
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

    let mut dist = conditional_dist(&dists, i, beta);

    for iter in 0..max_iters {
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

        dist = conditional_dist(&dists, i, beta);
    }

    (dist, beta)
}

fn symmetrised_dist_search_serial(dists: &DistanceMatrix, target_perp: f64) -> Joint {
    let n = dists.shape()[0] as usize;
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        let (p_ji, _beta) = perp_search(&dists, i, MAX_PERP_SEARCH_ITERS, target_perp);
        result.row_mut(i).assign(&p_ji);
    }
    (&result + &result.t()) / (2.0f64 * (n as f64))
}

fn symmetrised_dist_search(dists: &DistanceMatrix, target_perp: f64) -> Joint {
    let n = dists.shape()[0] as usize;

    let mut vs = (0..n).into_par_iter().map(|i| {
        let (p_ji, _beta) = perp_search(&dists, i, MAX_PERP_SEARCH_ITERS, target_perp);
        p_ji.to_vec()
    }).flatten().collect();

    let result = Array2::from_shape_vec((n, n), vs).unwrap();
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

fn grad(proj: &Array2<f64>, p_ij: &Joint, q_ij: &Joint,
        lo_dists: &DistanceMatrix)
        -> Array2<f64> {
    let n = proj.shape()[0] as usize;
    // result is same size as proj
    // let mut result = Array::zeros(proj.raw_dim());
    let p_minus_q = p_ij.borrow() - q_ij.borrow();

    // for i in 0..n {
    let d = (0..n).into_par_iter().map(|i| {
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

        sum.to_vec()
    }).flatten().collect();
    // println!("d {:?}", d);
    let mut result = Array::from_shape_vec((n, 2), d).unwrap();
    //result.row_mut(i).assign(&sum);
    result + &(proj * 0.00001)
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

fn convert(n: u8) -> u32 {
    let v: u32 = n.into();
    0xff000000 | (v << 16) | (v << 8) | v
}

fn update_proj(buf: &mut Vec<u32>, proj: &Array2<f64>, lbls: &Vec<u8>, maybe_imgs: Option<&Array2<u8>>) {
    let mut rng = rand::thread_rng();

    unsafe {
        libc::memset(buf.as_mut_ptr() as _, 0, buf.len() * mem::size_of::<u32>());
    }
    let xs = proj.slice(s![.., 0]);
    let ys = proj.slice(s![.., 1]);
    let min_x = vmin(&xs);
    let max_x = vmax(&xs);
    let min_y = vmin(&ys);
    let max_y = vmax(&ys);

    for (i, pt) in proj.outer_iter().enumerate() {
        let px = (pt[0] - min_x) / (max_x - min_x);
        let py = (pt[1] - min_y) / (max_y - min_y);

        let xx = (px * WIDTH as f64) as usize;
        let yy = (py * HEIGHT as f64) as usize;

        let lbl_colour = MAP_COLOURS[lbls[i] as usize];
        if xx > 0 && yy > 0 && xx < WIDTH - 1 && yy < HEIGHT - 1 {
            buf[yy * WIDTH + xx - 1] = lbl_colour;
            buf[(yy - 1) * WIDTH + xx] = lbl_colour;
            buf[(yy + 1) * WIDTH + xx] = lbl_colour;
            buf[yy * WIDTH + xx + 1] = lbl_colour;
            buf[yy * WIDTH + xx] = lbl_colour;
        }

//        if rng.gen::<f64>() < 0.5 {
        if let Some(imgs) = maybe_imgs {
            let img = imgs.slice(s![i, ..]).into_shape((IMG_WIDTH, IMG_HEIGHT)).unwrap();
            if xx >= IMG_WIDTH && yy >= IMG_HEIGHT && xx < WIDTH - IMG_WIDTH && yy < HEIGHT - IMG_HEIGHT {
                for i in 0..IMG_HEIGHT {
                    for j in 0..IMG_WIDTH {
                        buf[(yy + i) * WIDTH + (xx + j)] = convert(img[[i, j]]);
                    }
                }
            }
        }
    }
}

fn main() {
    let (trn_size, rows, cols) = (TRN_SIZE, IMG_WIDTH, IMG_HEIGHT);

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
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

    let mut p_ij = symmetrised_dist_search(&dists, PERP) * SIMILARITY_EXAGG_FACTOR;

    for i in 0usize..MAX_SGD_ITERS {
        let lo_dists = distances(&proj);
        let q_ij = joint_t_dist(&lo_dists);

        println!("{}", i);
        if i == SIMILARITY_EXAGG_STOP_ITER {
            p_ij /= SIMILARITY_EXAGG_FACTOR;
        }

        let del: Array2<f64> = grad(&proj, &p_ij, &q_ij, &lo_dists);
        let velocity = LR * &del + 0.9 * &velocity_prev;
        velocity_prev.assign(&velocity);
        proj = &proj - &velocity;

        let avg: Array1<f64> = proj.sum_axis(Axis(0)) / proj.len() as f64;
        proj = &proj - &avg;

        update_proj(&mut buf, &proj, &trn_lbl, if SHOW_IMGS && i >= SIMILARITY_EXAGG_STOP_ITER {
            Some(&u8_images)
        } else {
            None
        });
        window.update_with_buffer_size(&buf, WIDTH, HEIGHT).unwrap();
    }
}
