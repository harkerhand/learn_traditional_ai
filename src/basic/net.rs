use crate::basic::tensor_diff::{NdTensor, SharedTensor, Tensor, add, mat_mul, relu};
use ndarray::IxDyn;
use std::cell::RefCell;
use std::rc::Rc;

pub struct MlpNet {
    pub weights: Vec<SharedTensor>,
    pub biases: Vec<SharedTensor>,
}

impl MlpNet {
    pub fn new(dims: &[usize]) -> MlpNet {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for window in dims.windows(2) {
            let weight =
                tch::Tensor::randn(&[window[0] as i64, window[1] as i64], tch::kind::FLOAT_CPU)
                    * (2.0 / window[0] as f64).sqrt();
            let weight = Rc::new(RefCell::new(Tensor::from(weight)));
            let bias = Tensor::new(NdTensor::zeros(IxDyn(&[window[1]])));
            weights.push(weight);
            biases.push(bias);
        }

        MlpNet { weights, biases }
    }

    pub fn forward(&self, input: SharedTensor) -> SharedTensor {
        let mut x = input;
        for (i, (weight, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            x = mat_mul(x, Rc::clone(weight));
            x = add(x, Rc::clone(bias));
            if i != self.weights.len() - 1 {
                x = relu(x);
            }
        }
        x
    }

    pub fn parameters(&self) -> Vec<SharedTensor> {
        self.weights
            .iter()
            .cloned()
            .chain(self.biases.iter().cloned())
            .collect()
    }
}
