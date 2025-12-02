use crate::basic::tensor_diff::{NdTensor, SharedTensor};

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

    pub fn zero_grad(&mut self) {
        for param in &self.params {
            param.borrow_mut().grad.fill(0.0);
        }
    }

    pub fn step(&mut self) {
        for param in &self.params {
            let mut param_mut = param.borrow_mut();
            param_mut.data = &param_mut.data - &(&param_mut.grad * self.learning_rate);
        }
    }
}

pub struct AdamOptimizer {
    params: Vec<SharedTensor>,
    m: Vec<NdTensor>, // 存储每个参数的 m (一阶矩)
    v: Vec<NdTensor>, // 存储每个参数的 v (二阶矩)
    t: u64,           // 迭代次数
    // 超参数
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl AdamOptimizer {
    pub fn new(
        params: Vec<SharedTensor>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let m = params
            .iter()
            .map(|p| NdTensor::zeros(p.borrow().data.raw_dim()))
            .collect();
        let v = params
            .iter()
            .map(|p| NdTensor::zeros(p.borrow().data.raw_dim()))
            .collect();

        AdamOptimizer {
            params,
            m,
            v,
            t: 0,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }

    pub fn zero_grad(&mut self) {
        for param in &self.params {
            param.borrow_mut().grad.fill(0.0);
        }
    }

    pub fn step(&mut self) {
        self.t += 1;

        let eta = self.learning_rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.epsilon;

        // 计算偏差校正项
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            let mut p_mut = p.borrow_mut();
            let grad = &p_mut.grad; // g_t

            // 1. 更新 m_t
            self.m[i] = &self.m[i] * beta1 + grad * (1.0 - beta1);

            // 2. 更新 v_t
            let grad_sq = grad * grad; // 逐元素平方
            self.v[i] = &self.v[i] * beta2 + grad_sq * (1.0 - beta2);

            // 3. 偏差校正
            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // 4. 计算更新量
            // Update = eta * m_hat / (sqrt(v_hat) + epsilon)
            let update = m_hat.mapv(|x| eta * x) / (v_hat.mapv(f64::sqrt) + eps);

            // 5. 参数更新
            p_mut.data -= &update;
        }
    }
}
