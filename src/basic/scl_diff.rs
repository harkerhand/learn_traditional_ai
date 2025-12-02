use std::cell::RefCell;
use std::rc::Rc;

// 类型别名，用于简化 Value 的共享和可变引用
type SharedValue = Rc<RefCell<Value>>;

// 定义计算图节点的数据结构
pub struct Value {
    // 1. 值：当前计算结果
    pub data: f64,
    // 2. 梯度：损失 L 对该节点的导数 (dL/dValue)
    pub grad: f64,
    // 3. 依赖：指向输入节点（父节点）的列表
    pub children: Vec<SharedValue>,
    // 4. 后向函数：一个函数，定义了如何将梯度传给 children
    backward_fn: Option<Box<dyn Fn()>>,
}

impl Value {
    // 构造函数：创建一个叶子节点 (输入)
    pub fn new(data: f64) -> SharedValue {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
            backward_fn: None,
        }))
    }
}

pub fn add(a: SharedValue, b: SharedValue) -> SharedValue {
    let a_data = a.borrow().data;
    let b_data = b.borrow().data;
    let out = Value::new(a_data + b_data);
    out.borrow_mut().children.push(a.clone());
    out.borrow_mut().children.push(b.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        // 后向逻辑：dC/dA = 1, dC/dB = 1
        // 累加梯度：dL/dA += dL/dC * dC/dA
        let out_grad = out_clone.borrow().grad;
        a.borrow_mut().grad += out_grad * 1.0;
        b.borrow_mut().grad += out_grad * 1.0;
    }));

    out
}

pub fn mul(a: SharedValue, b: SharedValue) -> SharedValue {
    let a_data = a.borrow().data;
    let b_data = b.borrow().data;
    let out = Value::new(a_data * b_data);

    out.borrow_mut().children.push(a.clone());
    out.borrow_mut().children.push(b.clone());
    let out_clone = out.clone();
    out.borrow_mut().backward_fn = Some(Box::new(move || {
        // 后向逻辑：dC/dA = B, dC/dB = A
        // 累加梯度：dL/dA += dL/dC * dC/dA
        let out_grad = out_clone.borrow().grad;
        a.borrow_mut().grad += out_grad * b_data;
        b.borrow_mut().grad += out_grad * a_data;
    }));

    out
}

impl Value {
    // 后向传播函数
    pub fn backward(self_rc: &SharedValue) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        fn build_topo(
            v: SharedValue,
            topo: &mut Vec<SharedValue>,
            visited: &mut std::collections::HashSet<*const RefCell<Value>>,
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
        self_rc.borrow_mut().grad = 1.0;
        // 反向遍历拓扑排序的节点，执行后向函数
        for node in topo.into_iter().rev() {
            let node_ptr: *const _ = Rc::as_ptr(&node);
            if let Some(ref backward_fn) = node.borrow().backward_fn {
                backward_fn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_scl_diff() {
        let x = Value::new(2.0);
        let y = Value::new(3.0);

        let a = mul(x.clone(), x.clone()); // a = x * x
        assert_eq!(a.borrow().data, 4.0);
        let five = Value::new(5.0);
        let b = add(a.clone(), five.clone()); // b = a + 5
        assert_eq!(b.borrow().data, 9.0);
        let l = mul(b.clone(), y.clone()); // l = b * y
        assert_eq!(l.borrow().data, 27.0);
        Value::backward(&l);
        assert_eq!(l.borrow().grad, 1.0);  // dl/dl = 1
        assert_eq!(b.borrow().grad, 3.0);  // dl/db = y = 3
        assert_eq!(a.borrow().grad, 3.0);  // dl/da = dl/db * db/da = 3 * 1 = 3
        assert_eq!(five.borrow().grad, 3.0); // dl/d5 = dl/db * db/d5 = 3 * 1 = 3
        assert_eq!(x.borrow().grad, 12.0); // dl/dx = 2xy = 2 * 2 * 3 = 12
        assert_eq!(y.borrow().grad, 9.0);  // dl/dy = x*x + 5 = 4 + 5 = 9
    }
}


