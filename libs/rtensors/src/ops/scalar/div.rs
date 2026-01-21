use std::{ops::{Div, DivAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::{TensorValue, WeightValue}, TensorView, TensorViewMut}, grad::{self, GradNode}};

impl<'a, T, B> DivAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn div_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_div(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<'a, T, B> DivAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn div_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_div(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> DivAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn div_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_div(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
    }
}

impl<T, B> DivAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn div_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_div(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
    }
}

macro_rules! impl_div {
    ($type:ty) => {
        impl<'a, T, B> Div<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn div(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result /= rhs;
                result
            }
        }

        impl<'a, T, B> Div<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn div(self, rhs: &T) -> Self::Output {
                let mut result = self.owned();
                result /= rhs;
                result
            }
        }
    };
}

impl_div!(&TensorViewMut<'a, T, B>);
impl_div!(TensorViewMut<'a, T, B>);
impl_div!(&TensorView<'a, T, B>);
impl_div!(TensorView<'a, T, B>);
impl_div!(&TensorBase<T, B>);
impl_div!(TensorBase<T, B>);

impl<T, B> std::ops::Div<T> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    #[grad::when_enabled(ctx)]
    fn div(self, rhs: T) -> Self::Output {
        self.borrow_mut().tensor /= rhs;
        let op = GradNode::DivScalar { // both are the same as addition of negative scalar
            input: self.node,
            scalar: rhs,
        };
        ctx.attach(
            self.inner.clone(),
            op
        )
    }
}

impl<T, B> std::ops::Div<T> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    #[grad::when_enabled(ctx)]
    fn div(self, rhs: T) -> Self::Output {
        self.borrow_mut().tensor /= rhs;
        let op = GradNode::DivScalar { // both are the same as addition of negative scalar
            input: self.node,
            scalar: rhs,
        };
        ctx.attach(
            self.inner.clone(),
            op
        )
    }
}