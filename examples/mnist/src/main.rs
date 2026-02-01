use std::path::PathBuf;

use rand::seq::SliceRandom;
use rtensors::{backend::{Backend, BackendMatMul}, core::{MetaTensorView, primitives::TensorBase, tensor::{AsView, RandomTensor}, value::WeightValue}, grad::{self, optim::{Optim, SGD}}, ops::{broadcast::l1::mean_l1_loss, linalg::MatMul, unary::{UnaryOp}}};
#[cfg(feature = "cuda")]
use rtensors::backend::cuda::Cuda;

struct Layer<T: WeightValue, B: BackendMatMul<T>> {
    pub weight: TensorBase<T, B>,
    pub bias: TensorBase<T, B>,
}

struct DenseModel<T: WeightValue, B: BackendMatMul<T>> {
    pub layers: Vec<Layer<T, B>>,
}

impl<B: BackendMatMul<f32>> DenseModel<f32, B> {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let in_size = if i == 0 { input_size } else { hidden_size };
            let out_size = if i == num_layers - 1 { output_size } else { hidden_size };
            let weight = TensorBase::<f32, B>::uniform((in_size, out_size))
                .expect("Failed to create uniform tensor");
            let bias = TensorBase::<f32, B>::zeros((1, out_size));
            layers.push(Layer { weight, bias });
        }
        Self { layers }
    }

    fn forward(&self, mut x: TensorBase<f32, B>) -> TensorBase<f32, B> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = x.matmul(&layer.weight).unwrap() + &layer.bias;
            if i != self.layers.len() - 1 {
                x = x.relu();
            }
        }
        x.sigmoid()
    }

    fn register(&mut self, optim: &mut SGD<f32, B>) {
        for layer in &mut self.layers {
            optim.register_parameter(&mut layer.weight).unwrap();
            optim.register_parameter(&mut layer.bias).unwrap();
        }
    }
}

struct MnistDataset<B: BackendMatMul<f32>> {
    images: Vec<TensorBase<f32, B>>,
    targets: Vec<TensorBase<f32, B>>
}

struct MnistIter<'a, B: BackendMatMul<f32>> {
    images: &'a [TensorBase<f32, B>],
    targets: &'a [TensorBase<f32, B>],
    ordering: Vec<usize>,
    idx: usize,
}

impl<'a, B: BackendMatMul<f32>> MnistIter<'a, B> {
    fn new(dataset: &'a MnistDataset<B>) -> Self {
        let ordering: Vec<usize> = (0..dataset.images.len()).collect();
        let mut s = Self{
            images: &dataset.images,
            targets: &dataset.targets,
            ordering,
            idx: 0,
        };
        s.reset();
        s
    }
    fn reset(&mut self) {
        self.idx = 0;
        // reshuffle ordering
        let mut rng = rand::rng();
        self.ordering.shuffle(&mut rng);
    }
}

impl<'a, B: BackendMatMul<f32>> Iterator for MnistIter<'a, B> {
    type Item = (&'a TensorBase<f32, B>, &'a TensorBase<f32, B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.ordering.len() {
            return None;
        }
        let sample_idx = self.ordering[self.idx];
        self.idx += 1;
        Some((&self.images[sample_idx], &self.targets[sample_idx]))
    }
}

impl<B: BackendMatMul<f32>> MnistDataset<B> {
    fn load(path: PathBuf) -> Self {
        let mut images: Vec<TensorBase<f32, B>> = Vec::new();
        let mut targets: Vec<TensorBase<f32, B>> = Vec::new();
        // path points at a folder which holds folders: 0, 1, 2, 3, ..., 9 and in each is a bunch of images
        let subfolders = std::fs::read_dir(path).unwrap();
        for entry in subfolders {
            let entry = entry.unwrap();
            let label: u8 = entry.file_name().to_str().unwrap().parse().unwrap();
            let image_files = std::fs::read_dir(entry.path()).unwrap();
            for image_entry in image_files.into_iter().take(10) {
                let image_entry = image_entry.unwrap();
                let image = image::open(image_entry.path()).unwrap();
                let tensor: TensorBase::<u8, B> = image.into();
                let size = tensor.size();
                let img = tensor.into_dtype::<f32>()
                    .expect("Failed to convert tensor dtype")
                    .view_as((1, size))
                    .expect("Failed to flatten") / 255.0;
                images.push(img);
                let mut target: Vec<f32> = vec![0.; 10];
                target[label as usize] = 1.;
                targets.push(TensorBase::<_, B>::from_buf(target, (1, 10)).expect("Failed to create target tensor"));
            }
        }
        // Placeholder implementation
        Self {
            images,
            targets,
        }
    }
}

use rtensors::core::tensor::TensorAccess;

#[cfg(feature = "cuda")]
fn main () {
    let train_dset = MnistDataset::load(PathBuf::from("../data/mnist/training"));
    println!("Loaded {} training samples", train_dset.images.len());
    let size = train_dset.images[0].size();
    let mut model = DenseModel::new(size, 15, 10, 3);
    let mut optim = SGD::<f32, Cuda>::new(0.01);
    grad::with(|ctx| {
        model.register(&mut optim);

        // no batching yet so accumulate loss over multiple samples
        let mut nsamples = 0;
        let virtual_batch = 32;
        let epochs = 100;
        let mut viz = true;
        for _epoch in 0..epochs {
            let iterator = MnistIter::new(&train_dset);
            let mut total_loss = 0.0;
            let mut loss_samples = 0;
            for (x, y) in iterator.into_iter() {

                let input = x;
                let target = y;
    
                let out = model.forward(input.clone());
                let loss = mean_l1_loss(&out, &target);
                if viz {
                    let vis = ctx.graphviz(&loss).unwrap();
                    viz = false;
                    // save dot string to file
                    std::fs::write("mnist_loss_graph.dot", vis).expect("Failed to write graph");
                }

                total_loss += loss.item().expect("Failed to get loss item");
                loss_samples += 1;
                ctx.backwards::<f32, Cuda>(&loss).expect("Backwards failed");
                nsamples += 1;
                if nsamples >= virtual_batch {
                    optim.step().expect("Optimizer step failed");
                    nsamples = 0;
                }
            }
            println!("Epoch {}: Average Loss = {}", _epoch + 1, total_loss / loss_samples as f32);
        }
    });
}

#[cfg(not(feature = "cuda"))]
fn main() {
    use rtensors::backend::cpu::Cpu;

    let train_dset = MnistDataset::load(PathBuf::from("../data/mnist/training"));
    println!("Loaded {} training samples", train_dset.images.len());
    let size = train_dset.images[0].size();
    let mut model = DenseModel::new(size, 15, 10, 3);
    let mut optim = SGD::<f32, Cpu>::new(0.01);
    grad::with(|ctx| {
        model.register(&mut optim);

        // no batching yet so accumulate loss over multiple samples
        let mut nsamples = 0;
        let virtual_batch = 32;
        let epochs = 100;
        for _epoch in 0..epochs {
            let iterator = MnistIter::new(&train_dset);
            let mut total_loss = 0.0;
            let mut loss_samples = 0;
            for (x, y) in iterator.into_iter() {

                let input = x;
                let target = y;
    
                let out = model.forward(input.clone());
                let loss = mean_l1_loss(&out, &target);
                total_loss += loss.item().expect("Failed to get loss item");
                loss_samples += 1;
                ctx.backwards::<f32, Cpu>(&loss).expect("Backwards failed");
                nsamples += 1;
                if nsamples >= virtual_batch {
                    optim.step().expect("Optimizer step failed");
                    nsamples = 0;
                }
            }
            println!("Epoch {}: Average Loss = {}", _epoch + 1, total_loss / loss_samples as f32);
        }
    });
}