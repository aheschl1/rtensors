#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

#[cfg(feature = "cuda")]
/// Trait for types that can be stored in tensors (with CUDA support).
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TypeConstants +
    DeviceRepr +
    std::fmt::Debug +
    Send + Sync + 
    PartialEq + PartialOrd +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    std::ops::AddAssign +
    std::ops::SubAssign +
    std::ops::MulAssign +
    std::ops::Div<Output = Self> + 
    Absolute + 
    DTypeConversion +
    'static
{
    const DTYPE: crate::core::value::DType;
}

// a value which can be used for neural network weights
pub trait WeightValue : 
    rand::distr::uniform::SampleUniform + 
    TensorValue + 
    std::ops::Neg<Output=Self> + 
{
    fn from_f32(value: f32) -> Self;
    fn vexp(&self) -> Self;
    fn vexpm1(&self) -> Self;
    fn square_root(&self) -> Self;
    fn nat_log(&self) -> Self;
    fn nat_log1p(&self) -> Self;
    fn vfloor(&self) -> Self;
    fn vceil(&self) -> Self;
    fn vround(&self) -> Self;
    fn vtrunc(&self) -> Self;
    fn vlog(&self, base: Self) -> Self;
    fn vlog1p(&self, base: Self) -> Self;
    fn vsin(&self) -> Self;
    fn vcos(&self) -> Self;
    fn vtan(&self) -> Self;
    fn vasin(&self) -> Self;
    fn vacos(&self) -> Self;
    fn vatan(&self) -> Self;
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }
    fn eps_eq(&self, other: &Self, epsilon: f32) -> bool;
}

// FROM f32 IS A PLACEHOLDER FOR ADVANCED RANDOMNESS LOGIC LATER
impl WeightValue for f32 {
    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline(always)]
    fn vexp(&self) -> Self {
        self.exp()
    }

    #[inline(always)]
    fn vexpm1(&self) -> Self {
        self.exp_m1()
    }

    #[inline(always)]
    fn square_root(&self) -> Self {
        self.sqrt()
    }
    
    #[inline]
    fn eps_eq(&self, other: &Self, epsilon: f32) -> bool {
        (*self - *other).abs() < epsilon
    }

    #[inline(always)]
    fn nat_log(&self) -> Self {
        self.ln()
    }

    #[inline(always)]
    fn nat_log1p(&self) -> Self {
        self.ln_1p()
    }

    #[inline(always)]
    fn vfloor(&self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn vceil(&self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn vround(&self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn vtrunc(&self) -> Self {
        self.trunc()
    }
    
    #[inline(always)]
    fn vlog(&self, base: Self) -> Self {
        self.log(base)
    }

    #[inline(always)]
    fn vlog1p(&self, base: Self) -> Self {
        // (self + Self::ONE).log(base)
        self.ln_1p() / base.ln()
    }

    #[inline(always)]
    fn vsin(&self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn vcos(&self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn vtan(&self) -> Self {
        self.tan()
    }

    #[inline(always)]
    fn vasin(&self) -> Self {
        self.asin()
    }

    #[inline(always)]
    fn vacos(&self) -> Self {
        self.acos()
    }

    #[inline(always)]
    fn vatan(&self) -> Self {
        self.atan()
    }
}

impl WeightValue for f64 {
    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        value as f64
    }

    #[inline(always)]
    fn vexp(&self) -> Self {
        self.exp()
    }

    #[inline(always)]
    fn vexpm1(&self) -> Self {
        self.exp_m1()
    }

    #[inline(always)]
    fn square_root(&self) -> Self {
        self.sqrt()
    }
    
    #[inline]
    fn eps_eq(&self, other: &Self, epsilon: f32) -> bool {
        (*self - *other).abs() < epsilon as f64
    }
    #[inline(always)]
    fn nat_log(&self) -> Self {
        self.ln()
    }

    #[inline(always)]
    fn nat_log1p(&self) -> Self {
        self.ln_1p()
    }

    #[inline(always)]
    fn vfloor(&self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn vceil(&self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn vround(&self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn vtrunc(&self) -> Self {
        self.trunc()
    }

    #[inline(always)]
    fn vlog(&self, base: Self) -> Self {
        self.log(base)
    }

    #[inline(always)]
    fn vlog1p(&self, base: Self) -> Self {
        self.ln_1p() / base.ln()
    }

    #[inline(always)]
    fn vsin(&self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn vcos(&self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn vtan(&self) -> Self {
        self.tan()
    }

    #[inline(always)]
    fn vasin(&self) -> Self {
        self.asin()
    }

    #[inline(always)]
    fn vacos(&self) -> Self {
        self.acos()
    }

    #[inline(always)]
    fn vatan(&self) -> Self {
        self.atan()
    }
}

pub trait DTypeConversion {
    fn to_u8(&self) -> u8;
    fn to_i8(&self) -> i8;
    fn to_u16(&self) -> u16;
    fn to_i16(&self) -> i16;
    fn to_u32(&self) -> u32;
    fn to_i32(&self) -> i32;
    fn to_u64(&self) -> u64;
    fn to_i64(&self) -> i64;
    fn to_u128(&self) -> u128;
    fn to_i128(&self) -> i128;
    fn to_f32(&self) -> f32;
    fn to_f64(&self) -> f64;
    fn to_bool(&self) -> boolean;
    #[inline(always)]
    fn convert<N: TensorValue>(&self) -> N {
        unsafe{
            match N::DTYPE {
                DType::U8 => std::mem::transmute_copy(&self.to_u8()),
                DType::I8 => std::mem::transmute_copy(&self.to_i8()),
                DType::U16 => std::mem::transmute_copy(&self.to_u16()),
                DType::I16 => std::mem::transmute_copy(&self.to_i16()),
                DType::U32 => std::mem::transmute_copy(&self.to_u32()),
                DType::U128 => std::mem::transmute_copy(&self.to_u128()),
                DType::I32 => std::mem::transmute_copy(&self.to_i32()),
                DType::U64 => std::mem::transmute_copy(&self.to_u64()),
                DType::I64 => std::mem::transmute_copy(&self.to_i64()),
                DType::I128 => std::mem::transmute_copy(&self.to_i128()),
                DType::F32 => std::mem::transmute_copy(&self.to_f32()),
                DType::F64 => std::mem::transmute_copy(&self.to_f64()),
                DType::BOOL => std::mem::transmute_copy(&self.to_bool()),
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
/// Trait for types that can be stored in tensors.
/// 
/// Implemented for standard numeric types: f32, f64, i8-i128, u8-u128, isize, usize, bool.
pub trait TensorValue: 
    Copy + 
    Default +
    TypeConstants +
    Send + Sync +
    PartialEq + PartialOrd +
    std::fmt::Debug +
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> +
    std::ops::AddAssign +
    std::ops::SubAssign +
    std::ops::MulAssign +
    Absolute +
    DTypeConversion +
    std::ops::Div<Output = Self> + 
    'static
{
    const DTYPE: crate::core::value::DType;
}

/// Provides default constant values for tensor element types.
pub trait TypeConstants {
    const ZERO: Self;
    const ONE: Self;
    const MIN: Self;
    const MAX: Self;
    const TEST_TOLERANCE: Self;
}


macro_rules! impl_tensor_values {
    ($(($type:ty, $dtype:expr)),+ $(,)?) => {
        $(
            impl TensorValue for $type {
                const DTYPE: crate::core::value::DType = $dtype;
            }

            impl DTypeConversion for $type {
                #[inline(always)]
                fn to_u8(&self) -> u8 {
                    *self as u8
                }
                #[inline(always)]
                fn to_i8(&self) -> i8 {
                    *self as i8
                }
                #[inline(always)]
                fn to_u16(&self) -> u16 {
                    *self as u16
                }
                #[inline(always)]
                fn to_i16(&self) -> i16 {
                    *self as i16
                }
                #[inline(always)]
                fn to_u32(&self) -> u32 {
                    *self as u32
                }
                #[inline(always)]
                fn to_i32(&self) -> i32 {
                    *self as i32
                }
                #[inline(always)]
                fn to_u64(&self) -> u64 {
                    *self as u64
                }
                #[inline(always)]
                fn to_i64(&self) -> i64 {
                    *self as i64
                }
                #[inline(always)]
                fn to_u128(&self) -> u128 {
                    *self as u128
                }
                #[inline(always)]
                fn to_i128(&self) -> i128 {
                    *self as i128
                }
                #[inline(always)]
                fn to_f32(&self) -> f32 {
                    *self as f32
                }
                #[inline(always)]
                fn to_f64(&self) -> f64 {
                    *self as f64
                }
                #[inline(always)]
                fn to_bool(&self) -> boolean {
                    if *self != Self::ZERO {
                        boolean::TRUE
                    } else {
                        boolean::FALSE
                    }
                }
            }
        )+
    };
}

macro_rules! impl_default {
    ($type:ty, $zero:expr, $one:expr, $min:expr, $max:expr, $tolerance:expr) => {
        impl TypeConstants for $type {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const MIN: Self = $min;
            const MAX: Self = $max;
            const TEST_TOLERANCE: Self = $tolerance;
        }
    };
}

impl_tensor_values!(
    (f32, DType::F32), 
    (f64, DType::F64), 
    (i8, DType::I8), 
    (i16, DType::I16), 
    (i32, DType::I32), 
    (i64, DType::I64), 
    (i128, DType::I128), 
    // (isize, DType::I64), 
    (u8, DType::U8), 
    (u16, DType::U16), 
    (u32, DType::U32), 
    (u64, DType::U64), 
    (u128, DType::U128), 
    // (usize, DType::U64)
);

impl_default!(f32, 0.0f32, 1.0f32, f32::MIN, f32::MAX, 0.00001f32);
impl_default!(f64, 0.0f64, 1.0f64, f64::MIN, f64::MAX, 0.00001f64);
impl_default!(i8, 0i8, 1i8, i8::MIN, i8::MAX, 0i8);
impl_default!(i16, 0i16, 1i16, i16::MIN, i16::MAX, 0i16);
impl_default!(i32, 0i32, 1i32, i32::MIN, i32::MAX, 0i32);
impl_default!(i64, 0i64, 1i64, i64::MIN, i64::MAX, 0i64);
impl_default!(i128, 0i128, 1i128, i128::MIN, i128::MAX, 0i128);
// impl_default!(isize, 0isize, 1isize, isize::MIN, isize::MAX);
impl_default!(u8, 0u8, 1u8, u8::MIN, u8::MAX, 0u8);
impl_default!(u16, 0u16, 1u16, u16::MIN, u16::MAX, 0u16);
impl_default!(u32, 0u32, 1u32, u32::MIN, u32::MAX, 0u32);
impl_default!(u64, 0u64, 1u64, u64::MIN, u64::MAX, 0u64);
impl_default!(u128, 0u128, 1u128, u128::MIN, u128::MAX, 0u128);
// impl_default!(usize, 0usize, 1usize, usize::MIN, usize::MAX);

pub trait Absolute {
    fn absolute(&self) -> Self;
}

macro_rules! impl_absolute {
    ($($type:ty,)+) => {
        $(
            impl Absolute for $type {
                #[inline(always)]
                fn absolute(&self) -> Self {
                    (*self).abs()
                }
            }
        )+
    }
}
macro_rules! impl_absolute_ident {
    ($($type:ty,)+) => {
        $(
            impl Absolute for $type {
                #[inline(always)]
                fn absolute(&self) -> Self {
                    *self
                }
            }
        )+
    }
}

impl_absolute!(
    i8,
    i16,
    i32,
    i64,
    i128,
    f32,
    f64,
);

impl_absolute_ident!(
    u8,
    u16,
    u32,
    u64,
    u128,
    types::boolean,
);

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

use crate::core::value::types::boolean;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
#[repr(C)]
pub enum DType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    U128 = 5,
    I32 = 6,
    U64 = 7,
    I64 = 8,
    I128 = 9,
    F32 = 10,
    F64 = 11,
    BOOL = 12,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    U128(u128),
    I32(i32),
    U64(u64),
    I64(i64),
    I128(i128),
    F32(f32),
    F64(f64),
    BOOL(boolean),
}

impl Value {
    pub (crate) fn convert<T: TensorValue>(self) -> T{
        match self {
            Value::U8(v) => v.convert::<T>(),
            Value::I8(v) => v.convert::<T>(),
            Value::U16(v) => v.convert::<T>(),
            Value::I16(v) => v.convert::<T>(),
            Value::U32(v) => v.convert::<T>(),
            Value::U128(v) => v.convert::<T>(),
            Value::I32(v) => v.convert::<T>(),
            Value::U64(v) => v.convert::<T>(),
            Value::I64(v) => v.convert::<T>(),
            Value::I128(v) => v.convert::<T>(),
            Value::F32(v) => v.convert::<T>(),
            Value::F64(v) => v.convert::<T>(),
            Value::BOOL(v) => v.convert::<T>(),
        }
    }
}

impl<T> From<T> for Value where T: TensorValue {
    fn from(value: T) -> Self {
        match T::DTYPE {
            DType::U8 => Value::U8(value.to_u8()),
            DType::I8 => Value::I8(value.to_i8()),
            DType::U16 => Value::U16(value.to_u16()),
            DType::I16 => Value::I16(value.to_i16()),
            DType::U32 => Value::U32(value.to_u32()),
            DType::U128 => Value::U128(value.to_u128()),
            DType::I32 => Value::I32(value.to_i32()),
            DType::U64 => Value::U64(value.to_u64()),
            DType::I64 => Value::I64(value.to_i64()),
            DType::I128 => Value::I128(value.to_i128()),
            DType::F32 => Value::F32(value.to_f32()),
            DType::F64 => Value::F64(value.to_f64()),
            DType::BOOL => Value::BOOL(value.to_bool()),
        }
    }
}

#[allow(non_camel_case_types)]
pub mod types {
    use std::ops::{AddAssign, Deref, DerefMut};
    #[cfg(feature = "cuda")]
    use cudarc::driver::DeviceRepr;

    use crate::core::value::{DType, DTypeConversion, TensorValue, TypeConstants};

    #[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
    #[repr(C)]
    /// The boolean type for tensors.
    /// 
    /// # Operations
    /// - Addition (`+`): Logical OR operation.
    /// - Subtraction (`-`): Logical XOR operation.
    /// - Multiplication (`*`): Logical AND operation.
    pub struct boolean(pub bool);

    impl boolean {
        pub const FALSE: Self = Self(false);
        pub const TRUE: Self = Self(true);
    }

    // the poiunters and refs are for inner
    impl AsRef<bool> for boolean {
        #[inline(always)]
        fn as_ref(&self) -> &bool {
            &self.0
        }
    }

    impl AsMut<bool> for boolean {
        #[inline(always)]
        fn as_mut(&mut self) -> &mut bool {
            &mut self.0
        }
    }

    impl Deref for boolean {
        type Target = bool;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for boolean {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl From<u8> for boolean {
        #[inline(always)]
        fn from(value: u8) -> Self {
            Self(value != 0)
        }
    }

    impl TypeConstants for boolean {
        const ZERO: Self = Self::FALSE;
        const ONE: Self = Self::TRUE;
        const MIN: Self = Self::FALSE;
        const MAX: Self = Self::TRUE;
        const TEST_TOLERANCE: Self = Self::FALSE;
    }

    impl std::ops::Add for boolean {
        type Output = Self; // OR
        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 || rhs.0)
        }
    }

    impl std::ops::Sub for boolean {
        type Output = Self; // XOR
        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 != rhs.0)
        }
    }

    impl std::ops::Mul for boolean {
        type Output = Self; // AND
        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 && rhs.0)
        }
    }

    impl AddAssign for boolean {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0 || rhs.0;
        }
    }

    impl std::ops::SubAssign for boolean {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = self.0 != rhs.0;
        }
    }

    impl std::ops::MulAssign for boolean {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = self.0 && rhs.0;
        }
    }

    impl From<boolean> for bool {
        fn from(value: boolean) -> Self {
            value.0
        }
    }

    impl From<bool> for boolean {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl std::ops::Div for boolean {
        type Output = Self;
        fn div(self, _rhs: Self) -> Self::Output {
            todo!()
        }
    }

    impl TensorValue for boolean {
        const DTYPE: DType = DType::BOOL;
    }

    impl DTypeConversion for boolean {
        #[inline(always)]
        fn to_u8(&self) -> u8 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_i8(&self) -> i8 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_u16(&self) -> u16 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_i16(&self) -> i16 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_u32(&self) -> u32 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_i32(&self) -> i32 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_u64(&self) -> u64 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_i64(&self) -> i64 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_u128(&self) -> u128 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_i128(&self) -> i128 {
            if self.0 { 1 } else { 0 }
        }
        #[inline(always)]
        fn to_f32(&self) -> f32 {
            if self.0 { 1.0 } else { 0.0 }
        }
        #[inline(always)]
        fn to_f64(&self) -> f64 {
            if self.0 { 1.0 } else { 0.0 }
        }
        #[inline(always)]
        fn to_bool(&self) -> boolean {
            *self
        }
    }

    #[cfg(feature = "cuda")]
    unsafe impl DeviceRepr for boolean {}
}

#[cfg(test)]
mod tests {
    use crate::core::{value::types::boolean, Tensor};

    #[test]
    fn boolean_tensor_add() {
        let mut tensor = Tensor::<boolean>::zeros((2, 3));
        tensor += boolean(true);
        tensor += boolean(true);
        let expected = Tensor::<boolean>::ones((2, 3));
        assert_eq!(tensor, expected);
    }

    #[test]
    fn boolean_tensor_add_operation() {
        // Test OR operation (false + false = false, false + true = true, true + true = true)
        let a = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = a + b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(true), boolean(true)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn boolean_tensor_sub_operation() {
        // Test AND NOT operation (a && !b)
        let a = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = a - b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(true), boolean(false)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn boolean_tensor_mul_operation() {
        // Test AND operation (false * false = false, false * true = false, true * true = true)
        let a = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = a * b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(false), boolean(true)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn boolean_tensor_add_assign() {
        let mut a = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        a += b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        assert_eq!(a, expected);
    }

    #[test]
    fn boolean_tensor_sub_assign() {
        let mut a = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(true)
        ], (2,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        a -= b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false)
        ], (2,)).unwrap();
        assert_eq!(a, expected);
    }

    #[test]
    fn boolean_tensor_mul_assign() {
        let mut a = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(true), boolean(false)
        ], (3,)).unwrap();
        let b = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false), boolean(true)
        ], (3,)).unwrap();
        a *= b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false), boolean(false)
        ], (3,)).unwrap();
        assert_eq!(a, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_add() {
        use crate::core::primitives::CudaTensor;

        let mut tensor = CudaTensor::<boolean>::zeros((2, 3));
        tensor += boolean(true);
        tensor += boolean(true);
        let expected = CudaTensor::<boolean>::ones((2, 3));
        assert_eq!(tensor.cpu().unwrap(), expected.cpu().unwrap());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_add_operation() {
        use crate::core::primitives::CudaTensor;

        // Test OR operation (false + false = false, false + true = true, true + true = true)
        let a = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = (a + b).cpu().unwrap();
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(true), boolean(true)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_sub_operation() {
        use crate::core::primitives::CudaTensor;

        let a = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = (a - b).cpu().unwrap();
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(true), boolean(false)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_mul_operation() {
        use crate::core::primitives::CudaTensor;

        // Test AND operation (false * false = false, false * true = false, true * true = true)
        let a = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(true), boolean(true)
        ], (4,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true), boolean(false), boolean(true)
        ], (4,)).unwrap();
        let result = (a * b).cpu().unwrap();
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(false), boolean(false), boolean(true)
        ], (4,)).unwrap();
        assert_eq!(result, expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_add_assign() {
        use crate::core::primitives::CudaTensor;

        let mut a = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        a += b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        assert_eq!(a.cpu().unwrap(), expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_sub_assign() {
        use crate::core::primitives::CudaTensor;

        let mut a = CudaTensor::<boolean>::from_buf(vec![
            boolean(true), boolean(true)
        ], (2,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(false), boolean(true)
        ], (2,)).unwrap();
        a -= b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false)
        ], (2,)).unwrap();
        assert_eq!(a.cpu().unwrap(), expected);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn boolean_tensor_cuda_mul_assign() {
        use crate::core::primitives::CudaTensor;

        let mut a = CudaTensor::<boolean>::from_buf(vec![
            boolean(true), boolean(true), boolean(false)
        ], (3,)).unwrap();
        let b = CudaTensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false), boolean(true)
        ], (3,)).unwrap();
        a *= b;
        let expected = Tensor::<boolean>::from_buf(vec![
            boolean(true), boolean(false), boolean(false)
        ], (3,)).unwrap();
        assert_eq!(a.cpu().unwrap(), expected);
    }
}