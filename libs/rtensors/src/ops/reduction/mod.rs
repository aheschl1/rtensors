use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, TensorError}, value::{TensorValue, WeightValue}, MetaTensor, MetaTensorView, Shape}};

pub trait Accumulator<T: WeightValue> {
    fn initialize() -> Self;
    fn accumulate(&mut self, value: T);
    fn finalize(&self, count: usize) -> T;
}

pub struct AddAccumulator<T: WeightValue> {
    sum: T,
}

impl<T: WeightValue> Accumulator<T> for AddAccumulator<T> {
    #[inline(always)]
    fn initialize() -> Self { Self { sum: T::ZERO } }

    #[inline(always)]
    fn accumulate(&mut self, value: T) { self.sum += value;}

    #[inline(always)]
    fn finalize(&self, _: usize) -> T {
        self.sum
    }
}

pub struct ProdAccumulator<T: WeightValue> {
    prod: T,
}

impl<T: WeightValue> Accumulator<T> for ProdAccumulator<T> {
    #[inline(always)]
    fn initialize() -> Self { Self { prod: T::ONE } }

    #[inline(always)]
    fn accumulate(&mut self, value: T) { self.prod *= value;}

    #[inline(always)]
    fn finalize(&self, _: usize) -> T {
        self.prod
    }
}

pub struct MaxAccumulator<T: WeightValue> {
    max: T,
}

impl<T: WeightValue> Accumulator<T> for MaxAccumulator<T> {
    #[inline(always)]
    fn initialize() -> Self { Self { max: T::MIN } }

    #[inline(always)]
    fn accumulate(&mut self, value: T) {
        if value > self.max {
            self.max = value;
        }
    }

    #[inline(always)]
    fn finalize(&self, _: usize) -> T {
        self.max
    }
}

pub enum Accum<T: WeightValue> {
    Sum(AddAccumulator<T>),
    Prod(ProdAccumulator<T>),
    Max(MaxAccumulator<T>),
    Min(MinAccumulator<T>),
    Mean(MeanAccumulator<T>),
}

impl<T: WeightValue> Accumulator<T> for Accum<T> {
    #[inline(always)]
    fn accumulate(&mut self, value: T) {
        match self {
            Accum::Sum(acc) => acc.accumulate(value),
            Accum::Prod(acc) => acc.accumulate(value),
            Accum::Max(acc) => acc.accumulate(value),
            Accum::Min(acc) => acc.accumulate(value),
            Accum::Mean(acc) => acc.accumulate(value),
        }
    }

    #[inline(always)]
    fn finalize(&self, count: usize) -> T {
        match self {
            Accum::Sum(acc) => acc.finalize(count),
            Accum::Prod(acc) => acc.finalize(count),
            Accum::Max(acc) => acc.finalize(count),
            Accum::Min(acc) => acc.finalize(count),
            Accum::Mean(acc) => acc.finalize(count),
        }
    }

    #[inline(always)]
    fn initialize() -> Self {
        panic!("Cannot initialize Accum directly. Use specific accumulator types.")
    }
}

pub struct MinAccumulator<T: WeightValue> {
    min: T,
}

impl<T: WeightValue> Accumulator<T> for MinAccumulator<T> {
    #[inline(always)]
    fn initialize() -> Self { Self { min: T::MAX } }

    #[inline(always)]
    fn accumulate(&mut self, value: T) {
        if value < self.min {
            self.min = value;
        }
    }

    #[inline(always)]
    fn finalize(&self, _: usize) -> T {
        self.min
    }
}

pub struct MeanAccumulator<T: WeightValue> {
    sum: T,
}

impl<T: WeightValue> Accumulator<T> for MeanAccumulator<T> {
    #[inline(always)]
    fn initialize() -> Self { Self { sum: T::ZERO } }

    #[inline(always)]
    fn accumulate(&mut self, value: T) { self.sum += value;}

    #[inline(always)]
    fn finalize(&self, count: usize) -> T {
        self.sum / T::from_usize(count)
    }
}

pub enum ReductionOpTypes {
    Sum,
    Prod,
    Max,
    Min,
    Mean,
    Variance {
        unbiased: bool
    },
    Stdev {
        unbiased: bool
    },
    LogSumExp,
    Norm(NormType),
    ArgMax,
    ArgMin
}

impl ReductionOpTypes {
    #[inline(always)]
    pub const fn get_code(&self) -> u8 {
        match self {
            Self::Sum => 1,
            Self::Prod => 2,
            Self::Max => 3,
            Self::Min => 4,
            Self::Mean => 5,
            Self::Variance {..} | Self::Stdev { .. } => 6,
            Self::LogSumExp => 7,
            Self::Norm(_) => 8,
            Self::ArgMax => 9,
            Self::ArgMin => 10
            
        }
    }

    #[inline(always)]
    pub fn get_accumulator<T: WeightValue>(&self) -> Accum<T> {
        match self {
            Self::Sum => Accum::Sum(AddAccumulator::<T>::initialize()),
            Self::Prod => Accum::Prod(ProdAccumulator::<T>::initialize()),
            Self::Max => Accum::Max(MaxAccumulator::<T>::initialize()),
            Self::Min => Accum::Min(MinAccumulator::<T>::initialize()),
            Self::Mean => Accum::Mean(MeanAccumulator::<T>::initialize()),
            _ => panic!("Accumulator not implemented for this reduction type.")
        }
    }

    #[inline(always)]
    pub const fn initial_value<T: TensorValue>(&self) -> T {
        match self {
            Self::Sum => T::ZERO,
            Self::Prod => T::ONE,
            Self::Max => T::MIN,
            Self::Min => T::MAX,
            _ => panic!("Initial value not implemented for this reduction type.")
        }
    }
}

pub enum NormType {
    L1,
    L2
}

pub trait TotalReductionOp<T: TensorValue, B: Backend>: Sized + ReductionOp<T, B> {
    fn sum(&self) -> Result<TensorBase<T, B>, TensorError> {self.sum_at(Idx::Item)}
    fn prod(&self) -> Result<TensorBase<T, B>, TensorError> {self.prod_at(Idx::Item)}
    fn mean(&self) -> Result<TensorBase<T, B>, TensorError> {self.mean_at(Idx::Item)}
    fn max(&self) -> Result<TensorBase<T, B>, TensorError>{self.max_at(Idx::Item)}
    fn min(&self) -> Result<TensorBase<T, B>, TensorError>{self.min_at(Idx::Item)}
    fn var(&self) -> Result<TensorBase<T, B>, TensorError>{self.var_at(Idx::Item)}
    fn pop_var(&self) -> Result<TensorBase<T, B>, TensorError>{self.pop_var_at(Idx::Item)}
    fn std(&self, unbiased: bool) -> Result<TensorBase<T, B>, TensorError>{self.std_at(Idx::Item, unbiased)}
    fn norm(&self, norm: NormType) -> Result<TensorBase<T, B>, TensorError>{self.norm_at(Idx::Item, norm)}
    fn logsumexp(&self) -> Result<TensorBase<T, B>, TensorError>{self.logsumexp_at(Idx::Item)}
    fn l1_norm(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.norm_at(Idx::Item, NormType::L1)
    }
    fn l2_norm(&self) -> Result<TensorBase<T, B>, TensorError> {
        self.norm_at(Idx::Item, NormType::L2)
    }
    fn argmax(&self) -> Result<TensorBase<u64, B>, TensorError> {
        self.argmax_at(Idx::Item)
    }
    fn argmin(&self) -> Result<TensorBase<u64, B>, TensorError> {
        self.argmin_at(Idx::Item)
    }
}

pub trait ReductionOp<T: TensorValue, B: Backend> : Sized {
    
    fn sum_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn prod_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn mean_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn max_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn min_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn var_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn pop_var_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;
    fn std_at(&self, axes: impl Into<Idx>, unbiased: bool) -> Result<TensorBase<T, B>, TensorError>;
    fn norm_at(&self, axes: impl Into<Idx>, norm: NormType) -> Result<TensorBase<T, B>, TensorError>;
    fn logsumexp_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError>;

    fn l1_norm_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        self.norm_at(axes, NormType::L1)
    }

    fn l2_norm_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        self.norm_at(axes, NormType::L2)
    }

    fn argmax_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<u64, B>, TensorError>;
    fn argmin_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<u64, B>, TensorError>;
}

impl<T: TensorValue, B: Backend, V> TotalReductionOp<T, B> for V where V: ReductionOp<T, B>{}

#[inline(always)]
pub fn do_reduce<T, B>(
    op: ReductionOpTypes,
    axes: &Idx,
    tensor: &impl AsView<T, B>,
) -> Result<TensorBase<T, B>, TensorError>
where
    T: WeightValue,
    B: Backend,
{
    let tensor = tensor.view();
    match axes {
        Idx::Item => {
            let mut output = TensorBase::from_buf(vec![T::ZERO], vec![])?;
            tensor.backend.apply_reduce_contiguous_flat(
                tensor.buf,
                &mut output.buf,
                tensor.meta.offset,
                tensor.meta.size(),
                op,
            )?;
            Ok(output)
        }
        Idx::At(axis) => {
            let mut output =
                materialize_output::<T, B>(&tensor.meta, tensor.backend.clone(), axes)?;
            tensor.backend.apply_reduce(
                (tensor.buf, &tensor.meta),
                (&mut output.buf, &output.meta),
                *axis,
                op,
            )?;
            Ok(output)
        }
        _ => Err(TensorError::WrongDims(
            "Reduction over multiple axes is not implemented yet.".to_string(),
        )),
    }
}

#[inline(always)]
pub fn do_argmax<T, B>(
    op: ReductionOpTypes,
    axes: &Idx,
    tensor: &impl AsView<T, B>,
) -> Result<TensorBase<u64, B>, TensorError>
where
    T: WeightValue,
    B: Backend,
{
    let tensor = tensor.view();
    match axes {
        Idx::Item => {
            let mut output = TensorBase::<u64, _>::from_buf(vec![ 0u64 ], vec![])?;
            tensor.backend.apply_argmax_contiguous_flat(
                tensor.buf,
                &mut output.buf,
                tensor.meta.offset,
                tensor.meta.size(),
                op,
            )?;
            Ok(output)
        }
        Idx::At(axis) => {
            let mut output =
                materialize_output::<u64, B>(&tensor.meta, tensor.backend.clone(), axes)?;
            tensor.backend.apply_argmax(
                (tensor.buf, &tensor.meta),
                (&mut output.buf, &output.meta),
                *axis,
                op,
            )?;
            Ok(output)
        }
        _ => Err(TensorError::WrongDims(
            "Reduction over multiple axes is not implemented yet.".to_string(),
        )),
    }
}


impl<T: WeightValue, B: Backend, V> ReductionOp<T, B> for V
where
    V: AsView<T, B>,
{
    fn sum_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Sum, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Sum, &axes, &t)
        }
    }

    fn prod_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Prod, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Prod, &axes, &t)
        }
    }

    fn max_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Max, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Max, &axes, &t)
        }
    }

    fn min_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Min, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Min, &axes, &t)
        }
    }

    fn mean_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::Mean, &axes, &a)
        } else {
            do_reduce(ReductionOpTypes::Mean, &axes, &t)
        }
    }
    fn var_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::Variance { unbiased: true };
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(code, &axes, &a)
        }else {
            do_reduce(code, &axes, &t)
        }
    }
    fn pop_var_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::Variance { unbiased: false };
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(code, &axes, &a)
        }else {
            do_reduce(code, &axes, &t)
        }
    }
    fn std_at(&self, axes: impl Into<Idx>, unbiased: bool) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::Stdev { unbiased };
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(code, &axes, &a)
        }else {
            do_reduce(code, &axes, &t)
        }
    }
    fn logsumexp_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(ReductionOpTypes::LogSumExp, &axes, &a)
        }else {
            do_reduce(ReductionOpTypes::LogSumExp, &axes, &t)
        }
    }
    fn norm_at(&self, axes: impl Into<Idx>, norm: NormType) -> Result<TensorBase<T, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::Norm(norm);
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_reduce(code, &axes, &a)
        }else {
            do_reduce(code, &axes, &t)
        }
    }
    fn argmax_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<u64, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::ArgMax;
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_argmax(code, &axes, &a)
        }else {
            do_argmax(code, &axes, &t)
        }
    }
    fn argmin_at(&self, axes: impl Into<Idx>) -> Result<TensorBase<u64, B>, TensorError> {
        let axes = axes.into();
        let code = ReductionOpTypes::ArgMin;
        let t = self.view();
        if !t.is_contiguous() {
            let a = t.contiguous();
            do_argmax(code, &axes, &a)
        }else {
            do_argmax(code, &axes, &t)
        }
    }
    
}

#[inline]
fn materialize_output<T: TensorValue, B: Backend>(input: &MetaTensor, backend: B, axes: &Idx) -> Result<TensorBase<T, B>, TensorError>{
    let output_meta = reduction_output_meta(input.clone(), axes);
    let buf = backend.alloc(output_meta.size());

   
    Ok(TensorBase::from_parts(backend, buf?, output_meta, None))
}

#[inline]
fn reduction_output_meta(input: MetaTensor, axes: &Idx) -> MetaTensor {
    let mut output_shape = Vec::new();

    let idx_num = match axes {
        Idx::Item => 0,
        Idx::At(x) => *x,
        _ => panic!("Weird multi dimension reductions are not avaiable.")
    };
    
    for d in 0..input.rank() {
        if d == idx_num {
            // PyTorch and Numpy have a keep dim argument that removes this.
            output_shape.push(1);
        } else {
            output_shape.push(input.shape()[d]);

        }
    }
    let output_shape: Shape = Shape::from(output_shape);
    let strides = shape_to_stride(&output_shape);

    MetaTensor {
        shape: output_shape,
        strides,
        offset: 0
    }
}