use crate::basic::tensor_diff::SharedTensor;

pub struct SgdOptimizer {
    params: Vec<SharedTensor>,
    learning_rate: f64,
}

impl SgdOptimizer {
    pub fn new(params: Vec<SharedTensor>, learning_rate: f64) -> Self {
        SgdOptimizer {
            params,
            learning_rate,
        }
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.borrow_mut().grad.fill(0.0);
        }
    }

    pub fn step(&self) {
        for param in &self.params {
            let grad = &param.borrow().grad;
            let mut param_mut = param.borrow_mut();
            param_mut.data -= &(grad * self.learning_rate);
        }
    }
}