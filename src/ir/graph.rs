//! CompIR: The computation graph.
//!
//! A DAG of `Node`s where every node has a concrete op, fully-resolved input/output
//! shapes, and (after the kernel matching pass) an assigned kernel.

use std::collections::HashMap;

use crate::ir::ops::Op;
use crate::ir::types::*;

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// A single operation in the computation graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub extra_outputs: Vec<TensorId>,
    pub attrs: AttrMap,
    pub kernel: Option<KernelAssignment>,
    pub estimated_cost: Option<f64>,
}

/// A kernel assigned to a node after the matching pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelAssignment {
    pub kernel_id: KernelId,
    pub kernel_name: String,
    pub dispatch: DispatchConfig,
    pub path: ExecutionPath,
}

/// Execution path for ops that have prefill vs decode variants.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExecutionPath {
    Prefill,
    Decode,
    Unified,
    Dual {
        prefill_kernel: KernelId,
        prefill_kernel_name: String,
        prefill_dispatch: Box<DispatchConfig>,
        decode_kernel: KernelId,
        decode_kernel_name: String,
        decode_dispatch: Box<DispatchConfig>,
    },
}

/// GPU dispatch configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DispatchConfig {
    pub grid: [DispatchDim; 3],
    pub threadgroup: [u32; 3],
    pub shared_memory_bytes: u32,
}

/// A dispatch dimension that may depend on runtime parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DispatchDim {
    Static(u32),
    CeilDiv { param: ParamName, divisor: u32 },
    Expr(ParamExpr),
}

/// Simple expression for runtime-computed dispatch dimensions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ParamExpr {
    Param(ParamName),
    Const(u32),
    CeilDiv(Box<ParamExpr>, Box<ParamExpr>),
    Mul(Box<ParamExpr>, Box<ParamExpr>),
    Add(Box<ParamExpr>, Box<ParamExpr>),
    Min(Box<ParamExpr>, Box<ParamExpr>),
}

// ---------------------------------------------------------------------------
// Computation graph
// ---------------------------------------------------------------------------

/// The full computation graph for a model.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CompGraph {
    pub nodes: Vec<Node>,
    pub node_map: HashMap<NodeId, usize>,
    pub tensors: HashMap<TensorId, TensorRef>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub metadata: GraphMetadata,

    next_node_id: u32,
    next_tensor_id: u32,
}

/// Metadata about the compiled model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphMetadata {
    pub model_id: String,
    pub architecture: ArchitectureFamily,
    pub params: ModelParams,
    pub weight_files: Vec<String>,
    pub total_weight_bytes: usize,
    /// Absolute path to tokenizer.json in the HF cache (for runtime).
    pub tokenizer_path: Option<String>,
    /// BOS token id (1 for LLaMA/Mistral, model-specific for others).
    pub bos_token_id: Option<u32>,
    /// EOS token id — generation stops when this token is sampled.
    pub eos_token_id: Option<u32>,
    /// Absolute paths to each .safetensors shard for runtime mmap.
    pub weight_file_paths: Vec<String>,
    /// Weight quantization format for runtime inference (e.g., "f16", "q8_0", "q4_0").
    #[serde(default = "default_quant")]
    pub quant: String,
}

fn default_quant() -> String { "f16".to_string() }

/// Known model architecture families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArchitectureFamily {
    LLaMA,
    Mistral,
    Qwen,
    Phi,
    Gemma,
    GPTNeoX,
}

/// Model parameters extracted from config.json.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelParams {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RoPEScaling>,
    pub attention_type: AttentionType,
    pub ffn_type: FFNType,
    pub tie_word_embeddings: bool,
    /// Whether Q/K projections have per-head RMSNorm (Qwen3).
    #[serde(default)]
    pub qk_norm: bool,
}

/// RoPE scaling configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoPEScaling {
    pub scaling_type: String,
    pub factor: f64,
}

/// Attention mechanism variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AttentionType {
    MHA,
    GQA,
    MQA,
}

/// Feed-forward network variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FFNType {
    Standard,
    SwiGLU,
    GeGLU,
}

// ---------------------------------------------------------------------------
// Graph construction API
// ---------------------------------------------------------------------------

impl CompGraph {
    pub fn new(metadata: GraphMetadata) -> Self {
        CompGraph {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            tensors: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata,
            next_node_id: 0,
            next_tensor_id: 0,
        }
    }

    /// Create a new tensor and register it in the graph.
    pub fn new_tensor(&mut self, shape: Shape, dtype: DType, storage: StorageClass) -> TensorId {
        let id = TensorId(self.next_tensor_id);
        self.next_tensor_id += 1;
        self.tensors.insert(
            id,
            TensorRef { id, shape, dtype, storage, strides: None },
        );
        id
    }

    /// Add a node to the graph.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: Vec<TensorId>,
        output: TensorId,
        attrs: AttrMap,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            inputs,
            output,
            extra_outputs: Vec::new(),
            attrs,
            kernel: None,
            estimated_cost: None,
        });
        self.node_map.insert(id, idx);
        id
    }

    /// Add a node with multiple outputs (e.g., Split).
    pub fn add_node_multi_out(
        &mut self,
        op: Op,
        inputs: Vec<TensorId>,
        output: TensorId,
        extra_outputs: Vec<TensorId>,
        attrs: AttrMap,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            inputs,
            output,
            extra_outputs,
            attrs,
            kernel: None,
            estimated_cost: None,
        });
        self.node_map.insert(id, idx);
        id
    }

    /// Get a tensor by ID.
    pub fn tensor(&self, id: TensorId) -> &TensorRef {
        self.tensors.get(&id).expect("tensor not found in graph")
    }

    /// Get a mutable tensor by ID.
    pub fn tensor_mut(&mut self, id: TensorId) -> &mut TensorRef {
        self.tensors.get_mut(&id).expect("tensor not found in graph")
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> &Node {
        let idx = self.node_map[&id];
        &self.nodes[idx]
    }

    /// Get a mutable node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        let idx = self.node_map[&id];
        &mut self.nodes[idx]
    }

    /// Iterate nodes in topological order.
    pub fn topo_iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_compute_ops(&self) -> usize {
        self.nodes.iter().filter(|n| n.op.is_compute_bound()).count()
    }
}
