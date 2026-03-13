//! Core type system for the UNC IR.
//!
//! Every tensor in the graph has a fully-resolved shape, dtype, and storage class.
//! The key invariant: weight dimensions are always `Dim::Static` (known at compile time),
//! while sequence/batch dimensions are `Dim::Param` (known at runtime, but bounded).

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NodeId(pub u32);

/// Unique identifier for a tensor (output of a node or a weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorId(pub u32);

/// Unique identifier for a buffer in the memory plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BufferId(pub u32);

/// Unique identifier for a kernel in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct KernelId(pub u32);

// ---------------------------------------------------------------------------
// Dimension expressions
// ---------------------------------------------------------------------------

/// A dimension that is either known at compile time or parameterized.
///
/// Weight dimensions are always `Static`. Sequence length and batch size are `Param`
/// with known upper bounds — the compiler allocates for `max` and dispatches for actual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Dim {
    /// Fully resolved at compile time (e.g., hidden_dim = 4096).
    Static(usize),
    /// Resolved at runtime but bounded (e.g., seq_len ∈ [1, 8192]).
    Param(ParamDim),
}

impl Dim {
    /// Returns the compile-time value, or the max bound for parameterized dims.
    pub fn max_value(&self) -> usize {
        match self {
            Dim::Static(v) => *v,
            Dim::Param(p) => p.max,
        }
    }

    /// Returns the compile-time value, or the min bound for parameterized dims.
    pub fn min_value(&self) -> usize {
        match self {
            Dim::Static(v) => *v,
            Dim::Param(p) => p.min,
        }
    }

    /// Returns `Some(value)` if this dimension is statically known.
    pub fn as_static(&self) -> Option<usize> {
        match self {
            Dim::Static(v) => Some(*v),
            Dim::Param(_) => None,
        }
    }

    pub fn is_static(&self) -> bool {
        matches!(self, Dim::Static(_))
    }
}

/// A runtime-parameterized dimension with constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ParamDim {
    pub name: ParamName,
    pub min: usize,
    pub max: usize,
    /// Required alignment (e.g., 8 for SIMD). 1 means no alignment constraint.
    pub alignment: usize,
}

/// Named runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ParamName {
    SeqLen,
    TotalSeqLen, // seq_len + kv_cache_len (for attention)
    BatchSize,
}

impl fmt::Display for ParamName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamName::SeqLen => write!(f, "seq"),
            ParamName::TotalSeqLen => write!(f, "total_seq"),
            ParamName::BatchSize => write!(f, "batch"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// A tensor shape where each dimension is either static or parameterized.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns true if all dimensions are statically known.
    pub fn is_fully_static(&self) -> bool {
        self.0.iter().all(|d| d.is_static())
    }

    /// Returns the shape with all param dims replaced by their max bounds.
    pub fn max_shape(&self) -> Vec<usize> {
        self.0.iter().map(|d| d.max_value()).collect()
    }

    /// Total number of elements at maximum dimensions.
    pub fn max_numel(&self) -> usize {
        self.max_shape().iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            match d {
                Dim::Static(v) => write!(f, "{}", v)?,
                Dim::Param(p) => write!(f, "{}", p.name)?,
            }
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Tensor data types, including quantized formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    // Quantized types (block-structured)
    Q8_0,   // 8-bit: 32 elements per block, 1 x f16 scale
    Q4_0,   // 4-bit: 32 elements per block, 1 x f16 scale (symmetric)
    Q4_1,   // 4-bit: 32 elements per block, f16 scale + f16 min
    // Integer types (for indices, masks)
    I32,
    U32,
    Bool,
}

impl DType {
    /// Size in bytes per element (for non-quantized types).
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DType::F32 | DType::I32 | DType::U32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::Bool => Some(1),
            _ => None,
        }
    }

    /// For quantized types: number of elements per quantization block.
    pub fn block_num_elements(&self) -> Option<usize> {
        match self {
            DType::Q8_0 | DType::Q4_0 | DType::Q4_1 => Some(32),
            _ => None,
        }
    }

    /// For quantized types: size in bytes per block.
    pub fn block_size_bytes(&self) -> Option<usize> {
        match self {
            DType::Q8_0 => Some(34),   // 32 x i8 + 1 x f16
            DType::Q4_0 => Some(18),   // 16 bytes data + 2 bytes scale
            DType::Q4_1 => Some(20),   // 16 bytes data + 2+2 bytes scale/min
            _ => None,
        }
    }

    pub fn is_quantized(&self) -> bool {
        self.block_num_elements().is_some()
    }

    /// The accumulation dtype for compute involving this type.
    pub fn accum_dtype(&self) -> DType {
        DType::F32
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::Q8_0 => write!(f, "q8_0"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q4_1 => write!(f, "q4_1"),
            DType::I32 => write!(f, "i32"),
            DType::U32 => write!(f, "u32"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

// ---------------------------------------------------------------------------
// Storage classes
// ---------------------------------------------------------------------------

/// Where a tensor's data lives and how it's accessed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StorageClass {
    /// Model weight: mmap'd from safetensors/GGUF at a known byte offset.
    Weight(WeightBinding),

    /// Activation: computed at runtime, stored in a reusable buffer.
    Activation {
        buffer: Option<BufferId>,
    },

    /// KV cache: persistent across decode steps, pre-allocated for max_seq_len.
    KVCache {
        layer: usize,
        head_group: usize,
        role: KVRole,
    },

    /// Compile-time constant (e.g., RoPE frequency table, causal mask template).
    Constant {
        data: Vec<u8>,
    },
}

/// Whether a KV cache tensor holds keys or values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KVRole {
    Key,
    Value,
}

/// Binding a tensor to a specific location in the weight file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WeightBinding {
    /// Name in the safetensors file.
    pub name: String,
    /// Byte offset from the start of the weight data section.
    pub byte_offset: usize,
    /// Size in bytes.
    pub byte_size: usize,
    /// Original shape as stored in the file.
    pub file_shape: Vec<usize>,
    /// Original dtype as stored in the file.
    pub file_dtype: DType,
}

// ---------------------------------------------------------------------------
// Tensor reference
// ---------------------------------------------------------------------------

/// A fully-described tensor in the computation graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorRef {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
    pub storage: StorageClass,
    /// Memory layout: stride per dimension in elements.
    /// None means default row-major (C-contiguous).
    pub strides: Option<Vec<usize>>,
}

impl TensorRef {
    /// Total size in bytes at maximum dimensions.
    pub fn max_size_bytes(&self) -> usize {
        let numel = self.shape.max_numel();
        if let Some(elem_size) = self.dtype.element_size() {
            numel * elem_size
        } else {
            let block_elems = self.dtype.block_num_elements().unwrap();
            let block_bytes = self.dtype.block_size_bytes().unwrap();
            (numel / block_elems) * block_bytes
        }
    }

    /// Size in bytes at decode dimensions (Param dims use min value, i.e. seq_len=1).
    pub fn decode_size_bytes(&self) -> usize {
        let numel: usize = self.shape.0.iter().map(|d| d.min_value()).product();
        if let Some(elem_size) = self.dtype.element_size() {
            numel * elem_size
        } else {
            let block_elems = self.dtype.block_num_elements().unwrap();
            let block_bytes = self.dtype.block_size_bytes().unwrap();
            (numel / block_elems) * block_bytes
        }
    }
}

// ---------------------------------------------------------------------------
// Attributes for op-specific metadata
// ---------------------------------------------------------------------------

/// Op-specific attributes carried on graph nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Attr {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Bool(bool),
}

/// Convenience type for the attribute map.
pub type AttrMap = HashMap<String, Attr>;
