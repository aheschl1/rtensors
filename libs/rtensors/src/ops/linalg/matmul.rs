use crate::{backend::BackendMatMul, core::{Dim, MetaTensor, MetaTensorView, Shape, Strides, meta::ContiguityTypes, primitives::{OpTensor, TensorBase}, shape_to_stride, tensor::{AsTensor, AsView, TensorAccess, TensorError, seal}, untyped::AsUntypedTensor, value::TensorValue}, grad::{self, GradNode}, ops::linalg::MatMul};

// broadcasting state:
// does not broadcast batch dims, they must match exactly
impl<L, R, T, B> MatMul<R, T, B> for L
where
    T: TensorValue,
    B: BackendMatMul<T>,
    L: AsView<T, B> + seal::Sealed,
    R: AsView<T, B> + seal::Sealed,
{
    type Output = TensorBase<T, B>;
    // in progress. contiguity rules are as follows:
    // (ignore batch for a second)
    // 1. inner most dim of lhs (K) must be contiguous (stride=1)
    // 2. second inner most dim does NOT need to be contiguous
    // we will add arguments for lda, ldb, and ldc, which are acceptable from blas
    // these are the leading dimensions (strides) of the matrices (next row, or in terms of ldc, next batch)
    fn matmul(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError> {
        let lhs_view0 = self.view();
        let rhs_view0 = rhs.view();

        let mut _lhs_storage = None;
        let mut _rhs_storage: Option<TensorBase<T, B>> = None;

        if lhs_view0.rank() < 2 {
            return Err(TensorError::InvalidShape(
                "LHS tensor must have rank >= 2 for matmul".to_string(),
            ));
        }

        if rhs_view0.rank() < 2 {
            return Err(TensorError::InvalidShape(
                "RHS tensor must have rank >= 2 for matmul".to_string(),
            ));
        }

        let mut contiguity_type_lhs = contiguity_type(&lhs_view0.meta);
        let mut contiguity_type_rhs =  contiguity_type(&rhs_view0.meta);

        // materialize lhs to target contiguity if needed
        let lhs_view = if contiguity_type_lhs == ContiguityTypes::None {
            let c = lhs_view0.contiguous();
            _lhs_storage = Some(c);
            contiguity_type_lhs = ContiguityTypes::RowMajor; // now it is row major
            unsafe{_lhs_storage.as_ref().unwrap_unchecked().view()}
        } else {
            lhs_view0
        };
        // materialize rhs to target contiguity if needed
        let rhs_view = if contiguity_type_rhs == ContiguityTypes::None {
            let c = rhs_view0.contiguous();
            _rhs_storage = Some(c);
            contiguity_type_rhs = ContiguityTypes::RowMajor; // now it is row major
            unsafe{_rhs_storage.as_ref().unwrap_unchecked().view()}
        } else {
            rhs_view0
        };

        let lhs_meta = &lhs_view.meta;
        let rhs_meta = &rhs_view.meta;

        let lr = lhs_meta.rank();
        let rr = rhs_meta.rank();

        if lr != rr || lr < 2 {
            return Err(TensorError::InvalidShape(format!(
                "Both tensors must have the same rank >= 2, got lhs rank {} and rhs rank {}",
                lr, rr
            )));
        }

        // batch dims are all leading dims except the last two
        let lhs_batch_dims: Vec<usize> = lhs_meta.shape.0[..lr - 2].to_vec();
        let rhs_batch_dims: Vec<usize> = rhs_meta.shape.0[..rr - 2].to_vec();

        // let broadcasted_params = compute_broadcasted_params(&lhs_batch_meta, &rhs_batch_meta);

        // if broadcasted_params.is_err() {
        if lhs_batch_dims != rhs_batch_dims {
            return Err(TensorError::SizeMismatch(format!(
                "Batch dimensions must match for matmul, got lhs batch dims {:?} and rhs batch dims {:?}",
                lhs_batch_dims, rhs_batch_dims
            )));
        }

        let b = if lhs_batch_dims.is_empty() { // lhs matches rhs
            1
        } else {
            lhs_batch_dims.iter().product::<usize>()
        };

        // matrix dims: (..., M, K) @ (..., K, N)
        let m  = lhs_meta.shape[lr - 2];
        let k_l = lhs_meta.shape[lr - 1];
        let k_r = rhs_meta.shape[rr - 2];
        let n  = rhs_meta.shape[rr - 1];

        if k_l != k_r {
            return Err(TensorError::SizeMismatch(format!(
                "Inner matrix dimensions must match for matmul, got lhs K={} and rhs K={}",
                k_l, k_r
            )));
        }

        // -------- Output shape: (batch..., M, N) --------
        let mut out_shape_vec: Vec<Dim> = lhs_batch_dims;
        out_shape_vec.push(m);
        out_shape_vec.push(n);
        let out_shape: Shape = out_shape_vec.into();
        let out_strides = shape_to_stride(&out_shape);

        let mut buf = lhs_view.backend.alloc(b*n*m)?;

        lhs_view.backend.matmul(
            (lhs_view.buf, lhs_meta, contiguity_type_lhs),
            (rhs_view.buf, rhs_meta, contiguity_type_rhs),
            &mut buf,
            b,
            m,
            k_l,
            n
        )?;

        let result = TensorBase::from_parts(
            lhs_view.backend.clone(),
            buf,
            MetaTensor::new(out_shape.clone(), out_strides.clone(), 0),
            None
        );

        attach_matmul_grad::<T, B>(
            &result,
            lhs_view.contiguous(),
            rhs_view.contiguous(),
        );

        Ok(result)
    }

    fn dot(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError>
    {
        let lview = self.view();
        let rview = rhs.view();
        if lview.rank() != 1 || rview.rank() != 1 {
            return Err(TensorError::InvalidShape(
                "Dot product is only defined for 1-D tensors".to_string(),
            ));
        }

        // Perform the matmul.
        let mut m1 = lview.unsqueeze().matmul(unsafe { &rview.unsqueeze_at(1).unwrap_unchecked() })?;

        // Here we use squeeze in place to prevent two memcopys.
        m1.squeeze_inplace();
        Ok(m1)
    }

    fn outer(&self, rhs: &R) -> Result<TensorBase<T, B>, TensorError>
    {
        let lview = self.view();
        let rview = rhs.view();
        if lview.rank() != 1 || rview.rank() != 1 {
            return Err(TensorError::InvalidShape(
                "Outer product is only defined for 1-D tensors".to_string(),
            ));
        }

        // Outer product: a (M,) @ b (N,) -> (M, N)
        // We reshape a to (M, 1) and b to (1, N), then do matmul
        let lhs_reshaped = unsafe { lview.unsqueeze_at(1).unwrap_unchecked() };
        let rhs_reshaped = rview.unsqueeze();
        
        lhs_reshaped.matmul(&rhs_reshaped)
    }

}

#[inline(always)]
#[grad::if_enabled(ctx)]
fn attach_matmul_grad<T, B>(
    output: &TensorBase<T, B>,
    left: TensorBase<T, B>,
    right: TensorBase<T, B>,
) -> Option<()>
where
    T: TensorValue,
    B: BackendMatMul<T>,
{
    let op = GradNode::MatMul {
        left: left.op(),
        right: right.op(),
        left_input: left.as_untyped(),
        right_input: right.as_untyped(),
    };
    ctx.attach(output, op);
}

// we are only concerned with the last two dims for matmul
// this is because gemm expects one of the following:
// row major, in which the last dim in contiguous, and the rows can be strided
// column major, in which the second last dim is contiguous, and the columns can be strided
// everything else is "non-contiguous". In fact, the requirement is "at least one of the last two dims must be contiguous"
// 
// in fact, only column major is supported by blas; however, there are transpose tricks to make row major work as well
#[inline]
fn contiguity_type(
    meta: &MetaTensor,
) -> ContiguityTypes {
    let shape = &meta.shape;
    let strides = &meta.strides;

    if shape.len() < 2 {
        return ContiguityTypes::RowMajor;
    }

    // if strides[shape.len() - 1] != 1isize {
    //     return Ok(ContiguityTypes::None);
    // }

    // 2 cases: row major or column major
    // row major means -1 dim is contiguous
    let inner_contiguity = {
        if strides[shape.len() - 1] == 1isize {
            ContiguityTypes::RowMajor
        }
        // column major means -2 dim is contiguous
        else if strides[shape.len() - 2] == 1isize {
            ContiguityTypes::ColumnMajor
        } else {
            ContiguityTypes::None
        }
    };

    // we need to check on the batch dims - if there are multiple batch dims, they need to be contiguous together
    // if they are not, we say None, if they are, we take the previous
    let batch_meta = MetaTensor::new(
        Shape::from(&shape.0[..shape.len() - 2]), 
        Strides::from(&strides.0[..strides.len() - 2]), 
        0 // does not matter for contiguity check
    );
    if batch_meta.rank() > 1 {
        if !batch_meta.is_flat() {
            ContiguityTypes::None
        } else {
            inner_contiguity
        }
    } else {
        inner_contiguity
    }

}