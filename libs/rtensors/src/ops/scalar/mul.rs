use std::{ops::{Mul, MulAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::{TensorValue, WeightValue}, TensorView, TensorViewMut}, grad::{self, GradNode}};

impl<'a, T, B> MulAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_mul(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B> MulAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_mul(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> MulAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_mul(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> MulAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_mul(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_mul {
    ($type:ty) => {
        impl<'a, T, B> Mul<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn mul(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result *= rhs;
                result
            }
        }

        impl<'a, T, B> Mul<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn mul(self, rhs: &T) -> Self::Output {
                let mut result = self.owned();
                result *= rhs;
                result
            }
        }
    };
}

impl_mul!(&TensorViewMut<'a, T, B>);
impl_mul!(TensorViewMut<'a, T, B>);
impl_mul!(&TensorView<'a, T, B>);
impl_mul!(TensorView<'a, T, B>);
impl_mul!(&TensorBase<T, B>);
impl_mul!(TensorBase<T, B>);

impl<T, B> std::ops::Mul<T> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    #[grad::when_enabled(ctx)]
    fn mul(self, rhs: T) -> Self::Output {
        self.borrow_mut().tensor *= rhs;
        let op = GradNode::MulScalar { // both are the same as addition of negative scalar
            input: self.node,
            scalar: rhs,
        };
        ctx.attach(
            self.inner.clone(),
            op
        )
    }
}

impl<T, B> std::ops::Mul<T> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    #[grad::when_enabled(ctx)]
    fn mul(self, rhs: T) -> Self::Output {
        self.borrow_mut().tensor *= rhs;
        let op = GradNode::MulScalar { // both are the same as addition of negative scalar
            input: self.node,
            scalar: rhs,
        };
        ctx.attach(
            self.inner.clone(),
            op
        )
    }
}