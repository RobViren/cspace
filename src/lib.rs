use numpy::{
    ndarray::{Array1, Array2, Axis},
    IntoPyArray, PyArray2, PyReadonlyArray1,
};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

#[pyclass]
struct RustESN {
    res_size: usize,
    w_in: Array1<f32>,     // Input weights: [res_size]
    w_res: Array2<f32>,    // Reservoir weights: [res_size, res_size]
    leak_rates: Array1<f32>, // Logarithmically distributed leaks
    state: Array1<f32>,    // Current state
}

#[pymethods]
impl RustESN {
    #[new]
    fn new(
        seed: u64,
        res_size: usize,
        sparsity: f32,
        spectral_radius: f32,
        leak_min: f32,
        leak_max: f32,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // FIX: In rand 0.9+, Uniform::new returns a Result.
        // We explicitly use f32 to ensure the type inference is correct.
        let dist = Uniform::new(-1.0f32, 1.0f32).expect("Invalid distribution range");

        // 1. Initialize Input Weights (Dense, Uniform [-1, 1])
        let w_in = Array1::from_shape_fn(res_size, |_| dist.sample(&mut rng));

        // 2. Initialize Reservoir Weights (Sparse, Uniform [-1, 1])
        let mut w_res = Array2::<f32>::zeros((res_size, res_size));
        for i in 0..res_size {
            for j in 0..res_size {
                // rng.random() is the correct API for rand 0.9
                if rng.random::<f32>() < sparsity {
                    w_res[[i, j]] = dist.sample(&mut rng);
                }
            }
        }

        // 3. Spectral Radius Scaling (Power Iteration Method)
        let mut v = Array1::from_elem(res_size, 1.0);
        for _ in 0..20 {
            v = w_res.dot(&v);
            let norm = v.mapv(|x| x.powi(2)).sum().sqrt();
            if norm > 1e-6 {
                v /= norm;
            }
        }
        
        let v_next = w_res.dot(&v);
        let current_radius = v_next.mapv(|x| x.abs()).sum() / (v.mapv(|x| x.abs()).sum() + 1e-9);
        
        if current_radius > 0.0 {
            w_res *= spectral_radius / current_radius;
        }

        // 4. Logarithmic Leak Rates
        let log_min = leak_min.ln();
        let log_max = leak_max.ln();
        let leak_rates = Array1::from_shape_fn(res_size, |i| {
            let t = i as f32 / (res_size.saturating_sub(1).max(1)) as f32;
            (log_min + t * (log_max - log_min)).exp()
        });

        RustESN {
            res_size,
            w_in,
            w_res,
            leak_rates,
            state: Array1::zeros(res_size),
        }
    }

    /// Process a sequence of audio samples and return the history of states.
    fn forward_sequence<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
    ) -> Bound<'py, PyArray2<f32>> {
        let audio = audio.as_array();
        let seq_len = audio.len();
        
        let mut history = Array2::<f32>::zeros((seq_len, self.res_size));

        let w_in = &self.w_in;
        let w_res = &self.w_res;
        let leaks = &self.leak_rates;
        let mut current_state = self.state.clone();

        for (t, &sample) in audio.iter().enumerate() {
            let input_injection = w_in * sample;
            let recurrent_injection = w_res.dot(&current_state);
            
            let update = (input_injection + recurrent_injection).mapv(f32::tanh);

            // Optimized leaky update
            current_state = &current_state + leaks * (&update - &current_state);

            history.index_axis_mut(Axis(0), t).assign(&current_state);
        }

        self.state = current_state;
        history.into_pyarray(py)
    }

    fn reset_state(&mut self) {
        self.state.fill(0.0);
    }
}

#[pymodule]
fn espace_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustESN>()?;
    Ok(())
}