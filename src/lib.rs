use ndarray::prelude::*;
use pyo3::prelude::*;

use numpy::{IntoPyArray, PyArray2, PyArray3};

use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

#[pymodule]
fn fractal_ml(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "generate_direction_fractal")]
    pub fn generate_direction_fractal(
        num_samples: u64,
        step_scale: f64,
        directions: &PyArray3<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let len: usize = directions.shape()[0];
        let num_directions: usize = directions.shape()[1];
        let dim: usize = directions.shape()[2];

        let mut rng = thread_rng();
        let direction_dist = Uniform::from(0..num_directions);

        let mut output: Vec<f64> = vec![0.0; num_samples as usize * dim];

        for j in 0..(num_samples as usize) {
            let output_slice = &mut output[j * dim..(j + 1) * dim];
            for i in 0..len {
                let k: usize = direction_dist.sample(&mut rng);
                output_slice
                    .iter_mut()
                    .zip(
                        directions
                            .as_array()
                            .slice(s![i, k, ..])
                            .as_slice()
                            .unwrap(),
                    )
                    .for_each(|(a, b)| *a += b * step_scale.powi(i as i32));
            }
        }
        let py_output = Array2::from_shape_vec((num_samples as usize, dim), output).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(py_output.into_pyarray(py).to_owned())
    }

    Ok(())
}
