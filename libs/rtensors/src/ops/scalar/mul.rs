use std::{ops::{Mul, MulAssign}};

use crate::{backend::Backend, core::{TensorView, TensorViewMut, primitives::{OpTensor, TensorBase}, tensor::AsTensor, value::{TensorValue, WeightValue}}, grad::{self, GradNode}};


#[inline]
fn attach_mul_grad<T, B>(
    ctx: &grad::GradContext,
    view: &impl OpTensor,
    scalar: T,
)
where
    T: TensorValue,
    B: Backend,
{
    let node = view.op().unwrap_or_default();
    let op = GradNode::MulScalar {
        input: node,
        scalar: scalar.into()
    };
    ctx.attach(view, op);
}

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
        grad::when_enabled(|ctx| {
            attach_mul_grad::<T, B>(ctx, self, rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_mul_grad::<T, B>(ctx, self, *rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_mul_grad::<T, B>(ctx, self, rhs);
        });
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
        grad::when_enabled(|ctx| {
            attach_mul_grad::<T, B>(ctx, self, *rhs);
        });
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
