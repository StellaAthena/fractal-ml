use ndarray::prelude::*;
use pyo3::prelude::*;

use numpy::{IntoPyArray, PyArray2, PyArray3};

use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;



#[pymodule]
fn fractal_ml(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "generate_direction_fractal")]
    pub fn generate_direction_fractal(
        num_samples: u64,
        step_scale: f64,
        directions: &PyArray3<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let len = directions.shape()[0];
        let num_directions = directions.shape()[1];
        let dim = directions.shape()[2];

        let mut rng = thread_rng();
        let direction_dist = Uniform::from(0..num_directions);

        let mut output: Array2<f64> = Array2::zeros(Ix2(num_samples as usize, dim));

        for j in 0..(num_samples as usize) {
            let mut output_slice: ArrayViewMut1<f64> = output.slice_mut(s![j, ..]);
            for i in 0..len {
                let k: usize = direction_dist.sample(&mut rng);
                output_slice += &directions.as_array().slice(s![len - i - 1, k, ..]);
                output_slice *= step_scale;
            }
        }
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(output.into_pyarray(py).to_owned())
    }

    #[pyfn(m, "approx_box_counting")]
    pub fn approx_box_counting(
        scaling_factor: f64,
        min_scale: i32,
        max_scale: i32,
        dataset: &PyArray2<f64>,
    ) -> Vec<u64> {
        let len = dataset.shape()[0];

        let mut output = Vec::with_capacity(len);

        for j in 0..((max_scale - min_scale) as usize) {
            let mut hashes: HashSet<u64> = HashSet::new();
            let scale: f64 = scaling_factor.powi(j as i32 + min_scale);
            for i in 0..len {
                let mut hash_state = DefaultHasher::new();
                dataset.as_array()
                    .slice(s![i, ..])
                    .as_slice()
                    .unwrap()
                    .iter()
                    .for_each(|x| (x * scale).floor().to_bits().hash(&mut hash_state));
                hashes.insert(hash_state.finish());
            }
            output.push(hashes.len() as u64);
        }

        output
    }

    Ok(())
}
