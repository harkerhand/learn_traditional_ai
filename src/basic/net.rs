use crate::basic::tensor_diff::{add, mat_mul, relu, NdTensor, SharedTensor, Tensor};
use ndarray::IxDyn;
use std::cell::RefCell;
use std::rc::Rc;

pub struct MlpNet {
    pub fc1_weight: SharedTensor,
    pub fc1_bias: SharedTensor,
    pub fc2_weight: SharedTensor,
    pub fc2_bias: SharedTensor,
    pub fc3_weight: SharedTensor,
    pub fc3_bias: SharedTensor,
}

impl MlpNet {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> MlpNet {
        let fc1_weight = tch::Tensor::randn(&[input_size as i64, hidden_size as i64], tch::kind::FLOAT_CPU) * (2.0 / input_size as f64).sqrt();
        let fc1_weight = Rc::new(RefCell::new(Tensor::from(fc1_weight)));
        let fc1_bias = Tensor::new(NdTensor::zeros(IxDyn(&[hidden_size])));
        let fc2_weight = tch::Tensor::randn(&[hidden_size as i64, hidden_size as i64], tch::kind::FLOAT_CPU) * (2.0 / hidden_size as f64).sqrt();
        let fc2_weight = Rc::new(RefCell::new(Tensor::from(fc2_weight)));
        let fc2_bias = Tensor::new(NdTensor::zeros(IxDyn(&[hidden_size])));
        let fc3_weight = tch::Tensor::randn(&[hidden_size as i64, output_size as i64], tch::kind::FLOAT_CPU) * (2.0 / hidden_size as f64).sqrt();
        let fc3_weight = Rc::new(RefCell::new(Tensor::from(fc3_weight)));
        let fc3_bias = Tensor::new(NdTensor::zeros(IxDyn(&[output_size])));

        MlpNet {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        }
    }

    pub fn forward(&self, input: SharedTensor) -> SharedTensor {
        let matmul1 = mat_mul(input, self.fc1_weight.clone());
        let fc1 = add(matmul1, self.fc1_bias.clone());
        let fc1_act = relu(fc1);
        let matmul2 = mat_mul(fc1_act, self.fc2_weight.clone());
        let fc2 = add(matmul2, self.fc2_bias.clone());
        let fc2_act = relu(fc2);
        let matmul3 = mat_mul(fc2_act, self.fc3_weight.clone());
        let output = add(matmul3, self.fc3_bias.clone());
        output
    }

    pub fn parameters(&self) -> Vec<SharedTensor> {
        vec![
            self.fc1_weight.clone(),
            self.fc1_bias.clone(),
            self.fc2_weight.clone(),
            self.fc2_bias.clone(),
            self.fc3_weight.clone(),
            self.fc3_bias.clone(),
        ]
    }
}