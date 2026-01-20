use std::ops::{Index, IndexMut, Range};

use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, value::TensorValue, TensorViewMut}};

use super::primitives::TensorView;

pub type Dim = usize;

/// Represents the strides of a tensor (spacing between elements in each dimension).
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct Strides(pub Vec<isize>);

impl Strides {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, isize> {
        self.0.iter()
    }

    pub fn as_slice(&self) -> &[isize] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn remove(&mut self, index: usize) -> isize {
        self.0.remove(index)
    }

    pub fn squash_leading_dims(&self, n: usize) -> Strides {
        if n == 0 || self.is_empty() {
            return self.clone();
        }
        let mut new_strides = vec![];
        let mut prod = 1;
        for i in 0..n.min(self.len()) {
            prod *= self[i];
        }
        new_strides.push(prod);
        for i in n..self.len() {
            new_strides.push(self[i]);
        }
        Strides(new_strides)
    }
}

impl Index<usize> for Strides {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Strides {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl PartialEq<Vec<isize>> for Strides {
    fn eq(&self, other: &Vec<isize>) -> bool {
        &self.0 == other
    }
}

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

/// Represents the shape of a tensor (size of each dimension).
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct Shape(pub Vec<Dim>);


impl Shape {
    pub fn empty() -> Self {
        Shape(vec![])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Dim> {
        self.0.iter()
    }

    pub fn as_slice(&self) -> &[Dim] {
        &self.0
    }
    pub fn remove(&mut self, index: usize) -> Dim {
        self.0.remove(index)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, dim: &Dim) -> bool {
        self.0.contains(dim)
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    pub fn squash_leading_dims(&self, n: usize) -> Shape {
        if n == 0 || self.is_empty() {
            return self.clone();
        }
        let mut new_dims = vec![];
        let mut prod = 1;
        for i in 0..n.min(self.len()) {
            prod *= self[i];
        }
        new_dims.push(prod);
        for i in n..self.len() {
            new_dims.push(self[i]);
        }
        Shape(new_dims)
    }
}

impl Index<usize> for Shape {
    type Output = Dim;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl PartialEq<Vec<Dim>> for Shape {
    fn eq(&self, other: &Vec<Dim>) -> bool {
        &self.0 == other
    }
}

/// Tensor metadata containing shape, strides, and offset information.
/// 
/// This describes how to interpret a flat buffer as a multi-dimensional tensor.
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct MetaTensor {
    pub shape: Shape,
    pub strides: Strides,
    pub offset: usize,
}

impl From<&MetaTensor> for MetaTensor {
    fn from(meta: &MetaTensor) -> Self {
        meta.clone()
    }
}

impl MetaTensor {
    /// Creates tensor metadata with explicit shape, stride and offset.
    pub fn new(shape: impl Into<Shape>, strides: impl Into<Strides>, offset: usize) -> Self {
        Self { shape: shape.into(), strides: strides.into(), offset }
    }

    /// Returns true when the metadata describes a scalar (rank 0).
    pub fn is_scalar(&self) -> bool { self.strides.is_empty() }
    
    /// Returns true when the metadata describes a 1xN row tensor.
    pub fn is_row(&self) -> bool { self.shape.len() == 2 && self.shape[0] == 1 }
    
    /// Returns true when the metadata describes a 1-D column tensor.
    pub fn is_column(&self) -> bool { self.shape.len() == 1 }
    
    /// Number of dimensions (rank).
    pub fn dims(&self) -> usize { self.shape.len() }
    
    /// Total number of elements (product of all dimensions).
    pub fn size(&self) -> usize { self.shape.iter().product() }
    
    /// Whether the layout is contiguous in row-major order.
    pub fn is_contiguous(&self) -> bool { is_contiguous_relaxed(&self.shape, &self.strides) }

    /// a version of contiguity which allows for strides of grater than 1 on the innermost dimension
    /// this can be used for example in matmul to see if multiple batch dimensions are contiguous together
    /// is_contiguous could not be used in that case, because the metadata indicates a stride greater than 1 on the innermost batch dimension
    pub fn is_flat(&self) -> bool {
        is_flat(&self.shape, &self.strides)
    }
    
    /// Borrow the shape vector.
    pub fn shape(&self) -> &Shape { &self.shape }
    
    /// Borrow the stride vector.
    pub fn strides(&self) -> &Strides { &self.strides }
    
    /// Convert strides to byte strides given the item size.
    pub fn byte_strides(&self, item_size: usize) -> Vec<isize> {
        self.strides.iter().map(|s| s * item_size as isize).collect()
    }

    pub fn inner_dimensions(&self, dim: usize) -> usize {
        let mut count = 1;
        for i in 0..self.shape().len() {
            if i <= dim {
                continue;
            }
            count *= self.shape.as_slice()[i];
        }
        count.max(1)
    }

    pub fn outer_dimensions(&self, dim: usize) -> usize {
        if dim == 0 {
            return 1;
        }
        let mut count = 1;
        for i in 0..self.shape().len() {
        
            count *= self.shape.as_slice()[i];
            if i == dim - 1 {
                break;
            }
            
        }
        count

    }
    
    /// Return the starting offset (in elements) into the underlying buffer.
    pub fn offset(&self) -> usize { self.offset }
    
    /// Returns the size of a single dimension by index.
    pub fn dim(&self, dim: Dim) -> Dim { self.shape[dim] }
    
    /// Returns the number of dimensions (alias for `dims`).
    pub fn rank(&self) -> usize { self.shape.len() }
    
    /// Returns an iterator over all offsets in the underlying buffer for this tensor/view.
    pub fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let offset = self.offset;
        TensorOffsetIterator::new(self.shape.as_slice(), self.strides.as_slice(), offset)
    }
    
    /// Returns an iterator over all coordinate vectors.
    pub fn iter_coords(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        CoordIter::new(self.shape.as_slice())
    }

    /// Returns a vector of ranges representing contiguous memory blocks.
    /// The complexity is O(blocks) where blocks is the number of contiguous blocks.
    /// So, contiguous tensors return a single range in O(1) time.
    /// A matrix with three contiguous rows not sitting contiguously would return three ranges would be slower.
    /// 
    /// Note that nagative strides are treaterd as non-contiguous here. This makes it appropriate for memcpy use cases.
    pub(crate) fn iter_forward_contiguous_ranges(&self) -> Vec<Range<usize>> {
        let k = contiguous_suffix_len(&self.shape, &self.strides);
        if k == 0 {
            return self.iter_offsets().map(|o| o..o+1).collect();
        }
        let block_elems: usize = self.shape
            .iter()
            .rev()
            .take(k)
            .product();
        let n_blocks: usize = self.shape
            .iter()
            .take(self.shape.len() - k)
            .product();

        let mut ranges = Vec::with_capacity(n_blocks);

        let base = self.offset as isize;
        let outer_shape = &self.shape.as_slice()[..self.shape.len() - k];
        let outer_strides = &self.strides.as_slice()[..self.strides.len() - k];

        let mut idx = vec![0usize; outer_shape.len()];

        for _ in 0..n_blocks {
            let mut off = base;
            for (i, s) in idx.iter().zip(outer_strides) {
                off += *i as isize * *s;
            }

            let start = off as usize;
            ranges.push(start..start + block_elems);

            // increment outer index
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < outer_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }

        ranges

    }
    
    /// Returns the offset for the i-th element in row-major order.
    pub fn ith_offset(&self, i: usize) -> usize {
        let mut idx = i;
        let mut coords = vec![0; self.rank()];

        for d in (0..self.rank()).rev() {
            let dim = self.shape[d];
            coords[d] = idx % dim;
            idx /= dim;
        }

        let mut offset = self.offset as isize;
        for (c, stride) in coords.iter().zip(self.strides.iter()) {
            offset += *c as isize * *stride;
        }

        offset as usize
    }

    pub fn idx_to_offset(&self, idx: impl Into<Idx>) -> usize {
        let idx = idx.into();
        let idx: Vec<usize> = idx.into();
        let mut offset = self.offset as isize;
        for (c, stride) in idx.iter().zip(self.strides.iter()) {
            offset += *c as isize * *stride;
        }
        offset as usize
    }

    pub fn offset_to_idx(&self, offset: usize) -> Idx {
        let mut idx = vec![0; self.rank()];
        let mut rem = offset as isize - self.offset as isize;

        for d in 0..self.rank() {
            let stride = self.strides[d];
            if stride == 0 {
                idx[d] = 0;
                continue;
            }
            let mut c = rem / stride;
            let max = self.shape[d] as isize - 1;
            if c < 0 {
                c = 0;
            } else if c > max {
                c = max;
            }
            idx[d] = c as usize;
            rem -= c * stride;
        }

        // debug_assert!(
        //     self.idx_to_offset(&self.offset_to_idx(offset)) == offset
        // );

        idx.into()
    }

}

pub struct TensorOffsetIterator<'a> {
    shape:   &'a [usize],
    stride:  &'a [isize],
    offset0: isize,
    index:   Vec<usize>,
    started: bool,
    done:    bool,
}

impl<'a> TensorOffsetIterator<'a> {
    pub fn new(shape: &'a [usize], stride: &'a [isize], base_offset: usize) -> Self {
        let rank = shape.len();
        TensorOffsetIterator {
            shape,
            stride,
            offset0: base_offset as isize,
            index: vec![0; rank],
            started: false,
            done: false,
        }
    }
}

impl<'a> Iterator for TensorOffsetIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.shape.is_empty() {
            if self.started {
                self.done = true;
                return None;
            }
            self.started = true;
            return Some(self.offset0 as usize);
        }

        if !self.started {
            self.started = true;
        }

        let mut off = self.offset0;
        for (coord, stride) in self.index.iter().zip(self.stride.iter()) {
            off += (*coord as isize) * *stride;
        }
        debug_assert!(off >= 0, "Negative offset in ND iterator is illegal.");
        let phys = off as usize;

        for dim in (0..self.index.len()).rev() {
            self.index[dim] += 1;
            if self.index[dim] < self.shape[dim] {
                return Some(phys);
            }
            self.index[dim] = 0;
        }

        self.done = true;
        Some(phys)
    }
}

pub struct CoordIter<'a> {
    shape: &'a [usize],
    coords: Vec<usize>,
    started: bool,
    done: bool,
}

impl<'a> CoordIter<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        let r = shape.len();
        Self {
            shape,
            coords: vec![0; r],
            started: false,
            done: false,
        }
    }
}

impl<'a> Iterator for CoordIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.shape.is_empty() {
            if self.started {
                self.done = true;
                return None;
            }
            self.started = true;
            return Some(vec![]);
        }

        if !self.started {
            self.started = true;
            return Some(self.coords.clone());
        }

        // Multi-dimensional counting
        for d in (0..self.coords.len()).rev() {
            self.coords[d] += 1;
            if self.coords[d] < self.shape[d] {
                return Some(self.coords.clone());
            }
            self.coords[d] = 0;
        }

        self.done = true;
        // Some(self.coords.clone())
        None
    }
}


/// Computes the standard row-major stride for a given shape.
/// 
/// This calculates the strides needed for contiguous row-major memory layout,
/// where the last dimension has stride 1 and each earlier dimension's stride
/// is the product of all later dimensions' sizes.
#[inline(always)] // sits on matmul critical path
pub fn shape_to_stride(shape: &Shape) -> Strides {
    let mut stride: Vec<isize> = vec![1; shape.len()];
    for i in (0..shape.len()).rev() {
        if i < shape.len() - 1 {
            stride[i] = stride[i + 1] * shape[i + 1] as isize;
        }
    }
    Strides(stride)
}

#[inline(always)]
fn contiguous_suffix_len(shape: &Shape, strides: &Strides) -> usize {
    let mut expected = 1isize;
    let mut k = 0;

    for i in (0..shape.len()).rev() {
        if shape[i] <= 1 {
            continue;
        }
        if strides[i] != expected {
            break;
        }
        expected *= shape[i] as isize;
        k += 1;
    }
    k
}

/// Checks whether a layout (shape/stride) is contiguous in a relaxed sense:
/// ignores singleton dimensions and accepts empty shapes.
#[inline(always)]
pub(crate) fn is_contiguous_relaxed(shape: &Shape, stride: &Strides) -> bool {
    if shape.is_empty() { return true; }
    if shape.contains(&0) { return true; }
    if shape.len() != stride.len() { return false; }

    let mut expected = 1usize;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        let s = stride[i];
        if dim != 1 {
            if s.unsigned_abs() != expected { return false; }
            expected = expected.saturating_mul(dim);
        }
    }
    true
}

/// contiguity, but allowing for non 1 innermost stride
#[inline(always)]
fn is_flat(shape: &Shape, stride: &Strides) -> bool {
    if shape.is_empty() { return true; }
    if shape.contains(&0) { return true; }
    if shape.len() != stride.len() { return false; }

    let innermost_stride = stride[shape.len() - 1].unsigned_abs();
    let mut expected = innermost_stride;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        let s = stride[i];
        if dim != 1 {
            if s.unsigned_abs() != expected { return false; }
            expected = expected.saturating_mul(dim);
        }
    }
    true
}


/// Provides read-only access to tensor metadata (shape, strides, offset).
/// 
/// Implemented for `MetaTensor`, `TensorBase`, `TensorView`, and `TensorViewMut`.
pub trait MetaTensorView {

    /// Returns a reference to the underlying metadata.
    fn meta(&self) -> &MetaTensor;
    
    /// Borrow the shape vector.
    fn shape(&self) -> &Shape { &self.meta().shape }
    
    /// Borrow the stride vector.
    fn strides(&self) -> &Strides { &self.meta().strides }
    
    /// Convert strides to byte strides given the item size.
    fn byte_strides(&self, item_size: usize) -> Vec<isize> {
        self.meta().byte_strides(item_size)
    }
    
    /// Starting offset (in elements) into the underlying buffer.
    fn offset(&self) -> usize { self.meta().offset }
    
    /// Number of dimensions (rank).
    fn dims(&self) -> usize { self.meta().dims() }
    
    /// Size of one dimension by index.
    fn dim(&self, dim: Dim) -> Dim { self.meta().dim(dim) }
    
    /// Total number of elements (product of all dimensions).
    fn size(&self) -> usize { self.meta().size() }
    
    /// True for rank-0 (scalar) tensors.
    fn is_scalar(&self) -> bool { self.meta().is_scalar() }
    
    /// True for 1xN row tensors.
    fn is_row(&self) -> bool { self.meta().is_row() }
    
    /// True for 1-D column tensors.
    fn is_column(&self) -> bool { self.meta().is_column() }

    /// Number of dimensions (alias for `dims`).
    fn rank(&self) -> usize { self.meta().rank() }
    
    /// Whether the layout is contiguous in row-major order.
    fn is_contiguous(&self) -> bool { self.meta().is_contiguous() }

    /// Returns an iterator over all buffer offsets for this tensor/view.
    fn iter_offsets(&self) -> impl Iterator<Item = usize> + '_ { self.meta().iter_offsets() }

    fn iter_coords(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        self.meta().iter_coords()
    }

    /// Returns a vector of (dim_index, dim_size, dim_stride) for all non-singleton dimensions.
    fn non_singleton_dims(&self) -> Vec<(usize, Dim, isize)> {
        self.shape().iter()
            .zip(self.strides().iter())
            .enumerate()
            .filter(|(_, (&d, _))| d > 1)
            .map(|(i, (&d, &s))| (i, d, s))
            .collect()
    }
}

impl MetaTensorView for MetaTensor {
    fn meta(&self) -> &MetaTensor {
        self
    }
}

impl<B, T: TensorValue> MetaTensorView for TensorBase<T, B> 
where
    B: Backend,
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

impl<T: TensorValue, B> MetaTensorView for TensorView<'_, T, B>
where
    B: Backend,
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

impl <T: TensorValue, B: Backend> MetaTensorView for TensorViewMut<'_, T, B>
{
    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}


impl From<&Shape> for Shape {
    fn from(val: &Shape) -> Self {
        Shape(val.0.clone())
    }
}

impl From<Shape> for Vec<Dim> {
    fn from(val: Shape) -> Self {
        val.0
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(v: Vec<Dim>) -> Self {
        Shape(v)
    }
}

impl From<&[Dim]> for Shape {
    fn from(slice: &[Dim]) -> Self {
        Shape(slice.to_vec())
    }
}

impl AsRef<[Dim]> for Shape {
    fn as_ref(&self) -> &[Dim] {
        &self.0
    }
}

impl From<(Dim,)> for Shape {
    fn from(t: (Dim,)) -> Self {
        Shape(vec![t.0])
    }
}

impl From<(Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1])
    }
}

impl From<(Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2])
    }
}

impl From<(Dim, Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2, t.3])
    }
}

impl From<(Dim, Dim, Dim, Dim, Dim)> for Shape {
    fn from(t: (Dim, Dim, Dim, Dim, Dim)) -> Self {
        Shape(vec![t.0, t.1, t.2, t.3, t.4])
    }
}

impl From<Strides> for Vec<isize> {
    fn from(val: Strides) -> Self {
        val.0
    }
}

impl From<Vec<isize>> for Strides {
    fn from(v: Vec<isize>) -> Self {
        Strides(v)
    }
}

impl From<&[isize]> for Strides {
    fn from(slice: &[isize]) -> Self {
        Strides(slice.to_vec())
    }
}

impl AsRef<[isize]> for Strides {
    fn as_ref(&self) -> &[isize] {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum ContiguityTypes {
    RowMajor,
    ColumnMajor,
    None
}