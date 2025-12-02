use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Axis, Dim, IxDyn, IxDynImpl, OwnedRepr};
use std::cell::RefCell;
use std::rc::Rc;

pub type NdTensor = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
pub type SharedTensor = Rc<RefCell<Tensor>>;


pub struct Tensor {
    pub data: NdTensor,
    pub grad: NdTensor,
    pub children: Vec<SharedTensor>,
    backward_fn: Option<Box<dyn Fn()>>,
}

impl Tensor {
    pub fn new(data: NdTensor) -> SharedTensor {
        let grad = NdTensor::zeros(data.raw_dim());
        Rc::new(RefCell::new(Tensor {
            data,
            grad,
            children: Vec::new(),
            backward_fn: None,
        }))
    }
    pub fn argmax(&self) -> Vec<usize> {
        let data = &self.data;
        let mut result = Vec::with_capacity(data.shape()[0]);
        for row in data.axis_iter(Axis(0)) {
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;
            for (idx, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }
            result.push(max_idx);
        }
        result
    }
}

pub fn add(a: SharedTensor, b: SharedTensor) -> SharedTensor {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let out = Tensor::new(&a_data + &b_data);
    out.borrow_mut().children.push(a.clone());
    out.borrow_mut().children.push(b.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();
        a.borrow_mut().grad += &out_grad;
        let b_shape = b_data.raw_dim();
        if out_grad.raw_dim() != b_shape {
            let summed_grad = out_grad.sum_axis(ndarray::Axis(0));
            let reshaped_grad = summed_grad.into_shape_with_order(b_shape).unwrap();
            b.borrow_mut().grad += &reshaped_grad;
        } else {
            b.borrow_mut().grad += &out_grad;
        }
    }));

    out
}

pub fn mul(a: SharedTensor, b: SharedTensor) -> SharedTensor {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let out = Tensor::new(&a_data * &b_data);

    out.borrow_mut().children.push(a.clone());
    out.borrow_mut().children.push(b.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();
        a.borrow_mut().grad += &(&out_grad * &b_data);
        b.borrow_mut().grad += &(&out_grad * &a_data);
    }));

    out
}

pub fn mat_mul(a: SharedTensor, b: SharedTensor) -> SharedTensor {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let out = Tensor::new(a_data.dot(&b_data));

    out.borrow_mut().children.push(a.clone());
    out.borrow_mut().children.push(b.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();
        a.borrow_mut().grad += &out_grad.dot(&b_data.t());
        b.borrow_mut().grad += &a_data.t().dot(&out_grad);
    }));

    out
}

pub fn relu(a: SharedTensor) -> SharedTensor {
    let a_data = a.borrow().data.clone();
    let out_data = a_data.mapv(|x| if x > 0.0 { x } else { 0.0 });
    let out = Tensor::new(out_data);

    out.borrow_mut().children.push(a.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();
        let relu_grad = a_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        a.borrow_mut().grad += &(&out_grad * &relu_grad);
    }));

    out
}

pub fn mean(a: SharedTensor) -> SharedTensor {
    let a_data = a.borrow().data.clone();
    let mean_value = a_data.mean().unwrap();
    let out = Tensor::new(NdTensor::from_elem(IxDyn(&[]), mean_value));

    out.borrow_mut().children.push(a.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();
        let a_shape = a_data.raw_dim();
        let num_elements = a_data.len() as f64;
        let grad_contribution = out_grad[[]] / num_elements;
        a.borrow_mut().grad += &NdTensor::from_elem(a_shape, grad_contribution);
    }));
    out
}

pub fn log_softmax_cross_entropy(z: SharedTensor, y: SharedTensor) -> SharedTensor {
    // println!("{:?}, {:?}", z.borrow().data.raw_dim(), y.borrow().data.raw_dim());
    let z_data = z.borrow().data.clone();
    let y_data = y.borrow().data.clone();
    let batch_size = z_data.shape()[0] as f64;
    let z_max = z_data.fold_axis(Axis(1), f64::NEG_INFINITY, |&a, &b| a.max(b));
    let z_shifted = &z_data - &z_max.insert_axis(Axis(1));
    let exp_z = z_shifted.mapv(|x| x.exp());
    let sum_exp_z = exp_z.sum_axis(Axis(1)).insert_axis(Axis(1));

    let y_hat = exp_z / sum_exp_z;
    let log_y_hat = y_hat.mapv(|x| x.max(1e-12).ln());
    let loss_terms = &y_data * &log_y_hat;
    let sample_losses = loss_terms.sum_axis(Axis(1)).mapv(|x| -x);
    let mean_loss = sample_losses.mean().unwrap();
    let out = Tensor::new(NdTensor::from_elem(IxDyn(&[]), mean_loss));
    out.borrow_mut().children.push(z.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        let out_grad = out_clone.borrow().grad.clone();

        let z_grad = (&y_hat - &y_data) / batch_size;
        z.borrow_mut().grad += &(&z_grad * out_grad[[]]);
    }));
    out
}

impl Tensor {
    pub fn backward(self_rc: &SharedTensor) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        fn build_topo(
            v: SharedTensor,
            topo: &mut Vec<SharedTensor>,
            visited: &mut std::collections::HashSet<*const RefCell<Tensor>>,
        ) {
            let ptr: *const _ = Rc::as_ptr(&v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);

                // 遍历子节点的子节点（即输入节点）
                for child in v.borrow().children.iter() {
                    build_topo(Rc::clone(child), topo, visited);
                }
                topo.push(v);
            }
        }
        build_topo(self_rc.clone(), &mut topo, &mut visited);

        // 初始化输出节点的梯度为 1.0
        let dim = self_rc.borrow().data.raw_dim();
        self_rc.borrow_mut().grad = NdTensor::ones(dim);
        // 反向遍历拓扑排序的节点，执行后向函数
        for node in topo.into_iter().rev() {
            let node_ptr: *const _ = Rc::as_ptr(&node);
            if let Some(ref backward_fn) = node.borrow().backward_fn {
                backward_fn();
            }
        }
    }
}

impl From<tch::Tensor> for Tensor {
    fn from(value: tch::Tensor) -> Self {
        let kind = value.kind();
        assert_eq!(kind, tch::Kind::Float, "Only Float kind is supported");
        let size = value.numel();
        let mut vec = vec![0.0f32; size];
        value.copy_data(&mut vec, size);
        let array_d = ArrayBase::from_shape_vec(
            value.size().iter().map(|&d| d as usize).collect::<Vec<_>>(),
            vec.into_iter().map(|x| x as f64).collect(),
        )
            .unwrap();
        let dim = array_d.raw_dim();
        Tensor {
            data: array_d,
            grad: NdTensor::zeros(dim),
            children: Vec::new(),
            backward_fn: None,
        }
    }
}

pub fn to_one_hot(labels: &tch::Tensor, num_classes: i64) -> NdTensor {
    let labels = labels.to_kind(tch::Kind::Int64);
    let batch_size = labels.size()[0] as usize;
    let mut one_hot = NdTensor::zeros(IxDyn(&[batch_size, num_classes as usize]));
    for (i, class) in labels.iter::<i64>().unwrap().enumerate() {
        one_hot[[i, class as usize]] = 1.0;
    }
    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;
    #[test]
    fn test_tensor_diff() {
        let x_data = NdTensor::from_elem(IxDyn(&[2, 2]), 2.0);
        let y_data = NdTensor::from_elem(IxDyn(&[2, 2]), 3.0);
        let x = Tensor::new(x_data);
        let y = Tensor::new(y_data);

        let z = add(x.clone(), y.clone()); // z = x + y
        assert_eq!(z.borrow().data, NdTensor::from_elem(IxDyn(&[2, 2]), 5.0));
        let w = mul(z.clone(), y.clone()); // w = z * y
        assert_eq!(w.borrow().data, NdTensor::from_elem(IxDyn(&[2, 2]), 15.0));
        let l = relu(w.clone()); // out = relu(w)
        assert_eq!(l.borrow().data, NdTensor::from_elem(IxDyn(&[2, 2]), 15.0));

        Tensor::backward(&l);

        let l_grad = &l.borrow().grad; // dl/dl = 1
        assert_eq!(l_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 1.0));
        let w_grad = &w.borrow().grad; // dl/dw = 1 (since relu'(w) = 1 for w > 0)
        assert_eq!(w_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 1.0));
        let z_grad = &z.borrow().grad; // dl/dz = dl/dw * dw/dz = 1 * y = 3
        assert_eq!(z_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 3.0));
        let y_grad = &y.borrow().grad; // dl/dy = dl/dw * dw/dy = 1 * (x + 2*y) = 2 + 2*3 = 8
        assert_eq!(y_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 8.0));
        let x_grad = &x.borrow().grad; // dl/dx = dl/dz * dz/dx = 3 * 1 = 3
        assert_eq!(x_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 3.0));
    }

    #[test]
    fn test_real_eg() {
        let input_data = NdTensor::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let weights_data = NdTensor::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap();
        let input = Tensor::new(input_data);
        let weights = Tensor::new(weights_data);
        let logits = mat_mul(input.clone(), weights.clone()); // logits = input @ weights
        assert_eq!(logits.borrow().data, NdTensor::from_shape_vec(IxDyn(&[1, 2]), vec![-2.0, 2.0]).unwrap());
        let activated = relu(logits.clone()); // activated = relu(logits)
        assert_eq!(activated.borrow().data, NdTensor::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 2.0]).unwrap());
        Tensor::backward(&activated);
        let activated_grad = &activated.borrow().grad; // dactivated/dactivated = 1
        assert_eq!(activated_grad, &NdTensor::from_elem(IxDyn(&[1, 2]), 1.0));
        let logits_grad = &logits.borrow().grad; // dactivated/dlogits = [0, 1]
        assert_eq!(logits_grad, &NdTensor::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 1.0]).unwrap());
        let input_grad = &input.borrow().grad; // dactivated/dinput = logits_grad @ weights.T = [0, 1, 0, -1]
        assert_eq!(input_grad, &NdTensor::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 1.0, 0.0, -1.0]).unwrap());
        let weights_grad = &weights.borrow().grad; // dactivated/dweights = input.T @ logits_grad = [0, 1, 0, -2, 0, 3, 0, -4]
        assert_eq!(weights_grad, &NdTensor::from_shape_vec(IxDyn(&[4, 2]), vec![0.0, 1.0, 0.0, -2.0, 0.0, 3.0, 0.0, -4.0]).unwrap());
    }

    #[test]
    /// Z = X @ W + B
    /// L = mean(ReLU(Z) * M)
    fn test_complex() {
        let x_data = NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w_data = NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let b_data = NdTensor::from_shape_vec(IxDyn(&[2]), vec![-3.0, -2.0]).unwrap();
        let m_data = NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.0, 1.0, 0.5]).unwrap();
        let x = Tensor::new(x_data);
        let w = Tensor::new(w_data);
        let b = Tensor::new(b_data);
        let m = Tensor::new(m_data);
        let xw = mat_mul(x.clone(), w.clone()); // XW =  X @ W
        assert_eq!(xw.borrow().data, NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        let z = add(xw.clone(), b.clone()); // Z = XW + B
        assert_eq!(z.borrow().data, NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![-2.0, 0.0, 0.0, 2.0]).unwrap());
        let a = relu(z.clone()); // A = ReLU(Z)
        assert_eq!(a.borrow().data, NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 2.0]).unwrap());
        let c = mul(a.clone(), m.clone()); // C = A * M
        assert_eq!(c.borrow().data, NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 1.0]).unwrap());
        let l = mean(c.clone()); // mean(C)
        assert_eq!(l.borrow().data, NdTensor::from_elem(IxDyn(&[]), 0.25));
        Tensor::backward(&l);
        let l_grad = &l.borrow().grad; // dl/dl = 1
        assert_eq!(l_grad, &NdTensor::from_elem(IxDyn(&[]), 1.0));
        let c_grad = &c.borrow().grad; // dl/dc = 1/4
        assert_eq!(c_grad, &NdTensor::from_elem(IxDyn(&[2, 2]), 0.25));
        let a_grad = &a.borrow().grad; // dl/da = dl/dc * dc/da = 0.25 * M
        assert_eq!(a_grad, &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.125, 0.25, 0.25, 0.125]).unwrap());
        let z_grad = &z.borrow().grad; // dl/dz = dl/da * da/dz
        assert_eq!(z_grad, &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 0.125]).unwrap());
        let xw_grad = &xw.borrow().grad; // dl/dxw = dl/dz * dz/dxw = dl/dz
        assert_eq!(xw_grad, &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 0.125]).unwrap());
        let b_grad = &b.borrow().grad; // dl/db = dl/dz * dz/db = sum over batch of dl/dz
        assert_eq!(b_grad, &NdTensor::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.125]).unwrap());
        let x_grad = &x.borrow().grad; // dl/dx = dl/dxw * dxw/dx = dl/dxw @ W.T
        assert_eq!(x_grad, &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.0, 0.0, 0.125]).unwrap());
        let w_grad = &w.borrow().grad; // dl/dw = x.T @ dl/dxw
        assert_eq!(w_grad, &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.375, 0.0, 0.5]).unwrap());
    }

    #[test]
    fn test_softmax_ce() {
        let z_data = NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 2.0, -1.0, 1.0]).unwrap();
        let y_data = NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let z = Tensor::new(z_data);
        let y = Tensor::new(y_data);
        let l = log_softmax_cross_entropy(z.clone(), y.clone());
        assert!((l.borrow().data[[]] - 1.126928).abs() < 1e-6);
        Tensor::backward(&l);
        let l_grad = &l.borrow().grad;
        assert_eq!(l_grad, &NdTensor::from_elem(IxDyn(&[]), 1.0));
        let z_grad = &z.borrow().grad;
        println!("z_grad: {:?}", z_grad);
        assert!((z_grad - &NdTensor::from_shape_vec(IxDyn(&[2, 2]), vec![0.059602, -0.059602, -0.440399, 0.440399]).unwrap()).iter().all(|x| x.abs() < 1e-6));
    }
}