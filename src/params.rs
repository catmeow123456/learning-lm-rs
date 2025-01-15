use std::slice::from_raw_parts;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    // - safetensors里存储的是原始数据，你需要以FP32的形式读取出来，创建出项目所使用的张量。
    // - safetensors包含张量的形状，你无需对原始张量做任何变形。
    // - 当"tie_word_embeddings"属性被打开时，模型最开始以及最后的embedding矩阵数据相同，safetensors会只存储一份数据，我们测试用的story模型就是这样。作业阶段你可以只关心story模型，但是后续项目中你需要处理两个矩阵不同的情况。
    // - 你可以用`src/model.rs`中的测例检验你的实现是否正确。
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // "model.norm.weight", "lm_head.weight"
        // "model.layers.<i>.input_layernorm.weight"
        // "model.layers.<i>.self_attn.k_proj.weight"
        // "model.layers.<i>.mlp.gate_proj.weight"
        // "model.layers.<i>.self_attn.o_proj.weight"
        // "model.layers.<i>.self_attn.q_proj.weight"
        // "model.layers.<i>.self_attn.v_proj.weight"
        // "model.layers.<i>.mlp.down_proj.weight"
        // "model.layers.<i>.mlp.up_proj.weight"
        // "model.layers.<i>.post_attention_layernorm.weight"

        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape();
            let data = unsafe { from_raw_parts(tensor.data().as_ptr() as *const f32, shape.iter().product()) };
            let res = Tensor::new(data.to_vec(), &shape.to_vec());
            println!("{} = {:?}", name, res.shape());
            res
        };
        println!("{:?}", safetensor.names());
        let layer = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i))).collect(),
            wq: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i))).collect(),
            wk: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i))).collect(),
            wv: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i))).collect(),
            wo: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i))).collect(),
            rms_ffn_w: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i))).collect(),
            w_up: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i))).collect(),
            w_gate: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i))).collect(),
            w_down: (0..layer).map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i))).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
