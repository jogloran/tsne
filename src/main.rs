extern crate itertools;
extern crate mnist;
extern crate ndarray;
extern crate ndarray_linalg;

use itertools::Itertools;
use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Shape};
use ndarray::linalg::Dot;
use ndarray::prelude::*;

type Data = Array2<f32>;
// (ndatum, ndims)
type ProjectedData = Array3<f32>;
// (ndatum, x, y)
type DistanceMatrix = Array2<f32>; // (ndatum, ndatum)

fn distances(data: &Data) -> DistanceMatrix {
    let n: usize = data.shape()[0];
    Array2::from_shape_fn((n, n), |(i, j)| {
        let xi = data.slice(s![i, ..]);
        let xj = data.slice(s![j, ..]);
        norm_sq(&xi, &xj)
    })
}

fn conditional_dist(dists: &DistanceMatrix, i: usize, sigma_i: f32) -> Array1<f32> {
    let n = dists.shape()[0] as usize;
    let d = dists.slice(s![i, ..]);
    let mut all = Array1::from_shape_fn(n, |j| {
        (-d[j] / (2. * sigma_i * sigma_i)).exp()
    });
    all /= all.sum();
    all
}

fn perp_search(dists: &DistanceMatrix, data: &Data, i: usize, max_iters: u32, target_perp: f32) ->
(Array1<f32>, f32) {
    let mut s_min = 0.01;
    let mut s_max = 10.;
    let mut s = s_min + (s_max - s_min) / 2.;
    let mut iter = 0;
    let mut dist = conditional_dist(&dists, i, s);
    while iter < max_iters {
        println!("> {:?}", dist);
        let cand_perp = perp(&dist);
        if target_perp < cand_perp {
            s_max = s;
        } else {
            s_min = s;
        }
        s = s_min + (s_max - s_min) / 2.;
        iter += 1;
        println!("iter: {} s: {}, perp: {}", iter, s, cand_perp);

        dist = conditional_dist(&dists, i, s);
    }

    (dist, s)
}

fn conditional_dist_search(dists: &DistanceMatrix, data: &Data, target_perp: f32) {
    // for each datum x_i,
    // i is fixed
    data.outer_iter().enumerate().for_each(|(i, f)| {
        println!("f: {:?}", f.shape());
        let (p_ji, s_i) = perp_search(&dists, &data, i, 50, target_perp);
        println!("sigma = {} cand perp: {}", s_i, perp(&p_ji));
    })
}

fn H(dist: &Array1<f32>) -> f32 {
    let ref log_dist = dist.mapv(|x: f32| x.log2());
    -(dist * log_dist).sum()
}

fn perp(dist: &Array1<f32>) -> f32 {
    2.0_f32.powf(H(&dist))
}

fn grad(data: &Data, proj: &ProjectedData) {}

fn norm_sq(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.dot(b)
}

fn main() {
    let (trn_size, rows, cols) = (100, 28, 28);
    let size: i32 = trn_size * rows * cols;

    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize();

    let mut u8_images = Array::from_vec(trn_img)
        .into_shape((trn_size as usize, (rows * cols) as usize)).unwrap();
    let mut images = u8_images.mapv(|f| {
        (f as f32) / 255.
    });

    let mut distances = distances(&images);
    conditional_dist_search(&distances, &images, 2.0);
}
