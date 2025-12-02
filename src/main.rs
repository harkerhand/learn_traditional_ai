mod basic;

use crate::basic::log_softmax_cross_entropy;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use tch::nn::{Module, OptimizerConfig};
use tch::vision::dataset::Dataset;
use tch::Tensor;

const N_CLASSED: i64 = 10;
const LINEAR_RATE: f64 = 0.01;
const EPOCHS: i64 = 2;
const HIDDEN_SIZE: i64 = 4;

const INPUT_SIZE: i64 = 28 * 28;

#[derive(Debug)]
struct MlpNet {
    fc1: tch::nn::Linear,
    fc2: tch::nn::Linear,
    fc3: tch::nn::Linear,
}
impl MlpNet {
    fn new(vs: &tch::nn::Path) -> MlpNet {
        let fc1 = tch::nn::linear(
            vs / "fc1",
            INPUT_SIZE,
            HIDDEN_SIZE,
            Default::default(),
        );
        let fc2 = tch::nn::linear(
            vs / "fc2",
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            Default::default(),
        );
        let fc3 = tch::nn::linear(
            vs / "fc3",
            HIDDEN_SIZE,
            N_CLASSED,
            Default::default(),
        );
        MlpNet { fc1, fc2, fc3 }
    }
}

impl Module for MlpNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.view([-1, INPUT_SIZE]);
        let xs = xs.apply(&self.fc1).relu();
        let xs = xs.apply(&self.fc2).relu();
        xs.apply(&self.fc3)
    }
}


fn run() -> Result<(), Box<dyn std::error::Error>> {
    let device = tch::Device::cuda_if_available();
    let dataset = tch::vision::mnist::load_dir("data/mnist")?;
    let vs = tch::nn::VarStore::new(device);
    let net = MlpNet::new(&vs.root());
    let mut opt = tch::nn::Sgd::default().build(&vs, LINEAR_RATE)?;
    for epoch in 1..=EPOCHS {
        for (bimages, blabels) in dataset.train_iter(64).shuffle().to_device(device) {
            let logits = net.forward(&bimages);
            let loss = logits.cross_entropy_for_logits(&blabels);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        println!("Epoch: {}/{}", epoch, EPOCHS);
    }
    vs.save("mlp_model.ot")?;
    println!("Model saved to mlp_model.ot");

    let test_acc = test(&net, &dataset, device);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);
    Ok(())
}

fn test(net: &MlpNet, dataset: &Dataset, device: tch::Device) -> f64 {
    let test_size = dataset.test_images.size()[0];
    let mut sum_acc = 0.0;
    for (bimages, blabels) in dataset.test_iter(64).to_device(device) {
        let logits = net.forward(&bimages);
        let preds = logits.argmax(Some(1), false);
        let correct_preds = preds.eq_tensor(&blabels);
        let batch_acc = correct_preds.to_kind(tch::Kind::Float).mean(tch::Kind::Float).double_value(&[]);
        sum_acc += batch_acc * bimages.size()[0] as f64;
    }
    sum_acc / test_size as f64
}

fn myrun() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = tch::vision::mnist::load_dir("data/mnist")?;
    let net = basic::MlpNet::new(INPUT_SIZE as usize, HIDDEN_SIZE as usize, N_CLASSED as usize);
    let opt = basic::SgdOptimizer::new(net.parameters(), LINEAR_RATE);
    for epoch in 1..=EPOCHS {
        for (bimages, blabels) in dataset.train_iter(64).shuffle() {
            let logits = net.forward(Rc::new(RefCell::new(basic::Tensor::from(bimages))));
            let blabels = basic::to_one_hot(&blabels, N_CLASSED);
            let loss = log_softmax_cross_entropy(logits, basic::Tensor::new(blabels));
            opt.zero_grad();
            basic::Tensor::backward(&loss);
            opt.step();
        }
        println!("Epoch: {}/{}", epoch, EPOCHS);
    }
    println!("Training completed using custom implementation.");
    let test_acc = mytest(&net, &dataset);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);
    Ok(())
}

fn mytest(net: &basic::MlpNet, dataset: &Dataset) -> f64 {
    let test_size = dataset.test_images.size()[0];
    let mut sum_acc = 0.0;
    for (bimages, blabels) in dataset.test_iter(64) {
        let logits = net.forward(Rc::new(RefCell::new(basic::Tensor::from(bimages))));
        let preds = logits.borrow().argmax();
        let blabels = basic::to_one_hot(&blabels, N_CLASSED);
        let correct_preds = basic::Tensor::new(blabels).borrow().argmax();
        for (i, pred) in preds.iter().enumerate() {
            if *pred == correct_preds[i] {
                sum_acc += 1.0;
            }
        }
    }
    sum_acc / test_size as f64
}

fn main() {
    match myrun() {
        Ok(_) => println!("Training completed successfully."),
        Err(e) => eprintln!("Error during training: {}", e),
    }
}


