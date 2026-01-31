use std::{ops::{Div, DivAssign}};

use crate::{backend::Backend, core::{TensorView, TensorViewMut, primitives::{NodeOp, OpTensor, TensorBase}, tensor::AsTensor, value::{TensorValue, WeightValue}}, grad::{self, GradNode, NodeKey}};

#[inline]
fn attach_div_grad<T, B>(
    ctx: &grad::GradContext,
    view: &impl OpTensor,
    scalar: T,
)
where
    T: TensorValue,
    B: Backend,
{
    let node = view.op();
    let op = GradNode::DivScalar {
        input: node,
        scalar: scalar.into()
    };
    ctx.attach(view, op);
}

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
        grad::when_enabled(|ctx| {
            attach_div_grad::<T, B>(ctx, self, rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_div_grad::<T, B>(ctx, self, *rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_div_grad::<T, B>(ctx, self, rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_div_grad::<T, B>(ctx, self, *rhs);
        });
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