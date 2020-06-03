use pyo3::prelude::*;
use ndarray::prelude::*;

use numpy::{IntoPyArray, PyArray2, PyArray3};


use rand::thread_rng;
use rand::distributions::{Distribution,Uniform};

pub fn generate_direction_fractal(num_samples: u64,step_scale:f64,directions: &PyArray3<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let len = directions.shape()[0];
    let num_directions = directions.shape()[1];
    let dim = directions.shape()[2];

    let mut rng = thread_rng();
    let direction_dist = Uniform::from(0..num_directions);

    let mut output: Array2<f64> = Array2::zeros(Ix2(num_samples as usize, dim));

    for j in 0..(num_samples as usize) {
        for i in 0..len {
            let k: usize = direction_dist.sample(&mut rng);
            let direction: ArrayView1<f64> = directions.as_array().slice(s![i,k, ..]);
            let mut output_slice: ArrayViewMut1<f64> = output.slice_mut(s![j, ..]);
            output_slice += direction;
        }
    }
    let gil = GILGuard::acquire();
    let py = gil.python();
    Ok(output.into_pyarray(py).to_owned())
}