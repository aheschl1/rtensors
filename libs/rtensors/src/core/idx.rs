use crate::core::Dim;

/// Represents an index into a tensor.
/// 
/// Can be created from tuples, vectors, or single values.
/// 
/// # Examples
/// ```ignore
/// // From tuple
/// let idx: Idx = (0, 1, 2).into();
/// 
/// // From vector
/// let idx: Idx = vec![0, 1, 2].into();
/// 
/// // Using the coord! macro
/// let idx = coord![0, 1, 2];
/// 
/// // Single value
/// let idx: Idx = 5.into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Idx {
    /// Multi-dimensional coordinates
    Coord(Vec<Dim>),
    /// Single position in a 1D tensor
    At(usize),
    /// Scalar access (rank-0 tensor)
    Item
}

impl From<&Idx> for Idx {
    /// Clones an index reference into an owned index.
    fn from(value: &Idx) -> Self {
        match value {
            Idx::Coord(coords) => Idx::Coord(coords.clone()),
            Idx::At(i) => Idx::At(*i),
            Idx::Item => Idx::Item,
        }
    }
}

// Vec<Dim> implementation
impl From<Vec<Dim>> for Idx {
    fn from(value: Vec<Dim>) -> Self {
        Idx::Coord(value)
    }
}

impl From<&Vec<Dim>> for Idx {
    fn from(value: &Vec<Dim>) -> Self {
        Idx::Coord(value.clone())
    }
}

// Array slice implementation
impl From<&[Dim]> for Idx {
    fn from(value: &[Dim]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Fixed-size array implementations
impl<const N: usize> From<[Dim; N]> for Idx {
    fn from(value: [Dim; N]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Reference to fixed-size array
impl<const N: usize> From<&[Dim; N]> for Idx {
    fn from(value: &[Dim; N]) -> Self {
        Idx::Coord(value.to_vec())
    }
}

// Single usize (1D)
impl From<usize> for Idx {
    fn from(value: usize) -> Self {
        Idx::At(value)
    }
}

// Tuple implementations up to length 6
impl From<(usize,)> for Idx {
    fn from(value: (usize,)) -> Self {
        Idx::Coord(vec![value.0])
    }
}

impl From<(usize, usize)> for Idx {
    fn from(value: (usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1])
    }
}

impl From<(usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2])
    }
}

impl From<(usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3, value.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Idx {
    fn from(value: (usize, usize, usize, usize, usize, usize)) -> Self {
        Idx::Coord(vec![value.0, value.1, value.2, value.3, value.4, value.5])
    }
}

impl From<()> for Idx {
    fn from(_: ()) -> Self {
        Idx::Item
    }
}

impl Into<Vec<Dim>> for Idx {
    fn into(self) -> Vec<Dim> {
        match self {
            Idx::Coord(coords) => coords,
            Idx::At(i) => vec![i],
            Idx::Item => vec![],
        }
    }
}