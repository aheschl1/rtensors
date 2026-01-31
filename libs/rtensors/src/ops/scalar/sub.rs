use std::{ops::{Sub, SubAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, tensor::AsTensor, value::TensorValue, TensorView, TensorViewMut}, grad::{self, GradNode}};

#[inline]
fn attach_sub_grad<T, B>(
    ctx: &grad::GradContext,
    view: &impl crate::core::primitives::OpTensor,
)
where
    T: TensorValue,
    B: Backend,
{
    // subtraction by a scalar is equivalent to adding a negative scalar; the
    // backward for scalar-sub uses the same AddScalar node (no scalar stored)
    let node = view.op();
    let op = GradNode::AddScalar { input: node };
    ctx.attach(view, op);
}

impl<'a, T, B> SubAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_sub(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_sub_grad::<T, B>(ctx, self);
        });
    }
}

impl<'a, T, B> SubAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_sub(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_sub_grad::<T, B>(ctx, self);
        });
    }
}

impl<T, B> SubAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_sub(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_sub_grad::<T, B>(ctx, self);
        });
    }
}

impl<T, B> SubAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_sub(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_sub_grad::<T, B>(ctx, self);
        });
    }
}

macro_rules! impl_sub {
    ($type:ty) => {
        impl<'a, T, B> Sub<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result -= rhs;
                result
            }
        }

        impl<'a, T, B> Sub<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: &T) -> Self::Output {
                let mut result = self.owned();
                result -= rhs;
                result
            }
        }
    };
}

impl_sub!(&TensorViewMut<'a, T, B>);
impl_sub!(TensorViewMut<'a, T, B>);
impl_sub!(&TensorView<'a, T, B>);
impl_sub!(TensorView<'a, T, B>);
impl_sub!(&TensorBase<T, B>);
impl_sub!(TensorBase<T, B>);

