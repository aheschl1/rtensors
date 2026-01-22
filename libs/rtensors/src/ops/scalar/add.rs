use std::{ops::{Add, AddAssign}};

use crate::{backend::Backend, core::{primitives::TensorBase, value::TensorValue, TensorView, TensorViewMut}, grad::{self, GradNode}};
use crate::core::tensor::AsTensor;

#[inline]
fn attach_add_grad<T, B>(
    ctx: &grad::GradContext,
    view: &impl crate::core::primitives::OpTensor,
)
where
    T: TensorValue,
    B: Backend,
{
    let node = view.op().unwrap_or_default();
    let op = GradNode::AddScalar { input: node };
    ctx.attach(view, op);
}

impl<'a, T, B> AddAssign<T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_add(
            self.buf, 
            rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_add_grad::<T, B>(ctx, self);
        });
    }
}

impl<'a, T, B> AddAssign<&T> for TensorViewMut<'a, T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_add(
            self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_add_grad::<T, B>(ctx, self);
        });
    }
}

impl<T, B> AddAssign<T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: T) {
        self.backend.scalar_apply_add(
            &mut self.buf, 
            rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_add_grad::<T, B>(ctx, self);
        });
    }
}

impl<T, B> AddAssign<&T> for TensorBase<T, B> 
    where T: TensorValue,
          B: Backend,
{
    fn add_assign(&mut self, rhs: &T) {
        self.backend.scalar_apply_add(
            &mut self.buf, 
            *rhs,
            &self.meta
        ).unwrap();
        grad::when_enabled(|ctx| {
            attach_add_grad::<T, B>(ctx, self);
        });
    }
}

macro_rules! impl_add {
    ($type:ty) => {
        impl<'a, T, B> Add<T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: T) -> Self::Output {
                let mut result = self.owned();
                result += rhs;
                result
            }
        }

        impl<'a, T, B> Add<&T> for $type
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: &T) -> Self::Output {
                let mut result = self.owned();
                result += rhs;
                result
            }
        }
    };
}

impl_add!(&TensorViewMut<'a, T, B>);
impl_add!(TensorViewMut<'a, T, B>);
impl_add!(&TensorView<'a, T, B>);
impl_add!(TensorView<'a, T, B>);
impl_add!(&TensorBase<T, B>);
impl_add!(TensorBase<T, B>);
