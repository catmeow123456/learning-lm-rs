use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            // x = rms_norm(residual)
            // Q = RoPE(x @ Q_weight.T)
            // K = RoPE(x @ K_weight.T)
            // V = x @ V_weight.T
            // K = cat(K_cache, K)
            // V = cat(V_cache, V)
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            // score = Q @ K.T / sqrt(dim)
            // attn = softmax(score)
            // attn_V = attn @ V
            self_attention(&mut hidden_states, &mut att_scores,
                q.reshape(&vec![seq_len, self.n_q_h * self.dqkv]),
                full_k.reshape(&vec![total_seq_len, self.n_kv_h * self.dqkv]),
                full_v.reshape(&vec![total_seq_len, self.n_kv_h * self.dqkv]),
                self.n_kv_h, n_groups, seq_len, total_seq_len, self.dqkv);
            // out = attn_V @ O_weight.T
            // residual = out + residual
            OP::matmul_transb(&mut residual, 1.0, &hidden_states, &self.params.wo[layer], 1.0);
            // mlp
            mlp(&mut residual, &mut hidden_states, &mut gate_buf, &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        let mut cache = self.new_cache();
        let mut next_token: u32;
        let input = Tensor::<u32>::new(vec![self.bos_token_id], &vec![1]);
        let logits = self.forward(&input, &mut cache);
        next_token = OP::random_sample(&logits, top_p, top_k, temperature);
        for i in 0..token_ids.len() {
            let input = Tensor::<u32>::new(vec![token_ids[i]], &vec![1]);
            let logits = self.forward(&input, &mut cache);
            next_token = OP::random_sample(&logits, top_p, top_k, temperature);
        }
        for _ in 0..max_len {
            result.push(next_token);
            let input = Tensor::<u32>::new(vec![next_token], &vec![1]);
            let logits = self.forward(&input, &mut cache);
            next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            if next_token == self.eos_token_id {
                break;
            }
        }
        result
    }
}

// ``` python
// score = Q @ K.T / sqrt(dim)
// attn = softmax(score)
// attn_V = attn @ V
// ```
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    assert!(hidden_states.shape() == &vec![seq_len, n_kv_h * n_groups * dqkv]);
    assert!(att_scores.shape() == &vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    assert!(q.shape() == &vec![seq_len, n_kv_h * n_groups * dqkv]);
    assert!(k.shape() == &vec![total_seq_len, n_kv_h * dqkv]);
    assert!(v.shape() == &vec![total_seq_len, n_kv_h * dqkv]);
    let hidden = unsafe { hidden_states.data_mut() };
    let score = unsafe { att_scores.data_mut() };
    for i in 0..n_kv_h {
        let mut k_i = Tensor::<f32>::default(&vec![total_seq_len, dqkv]);
        for x in 0..total_seq_len {
            let dst_k_i = &mut unsafe { k_i.data_mut() }[x * dqkv..][..dqkv];
            dst_k_i.copy_from_slice(&k.data()[((x * n_kv_h + i) * dqkv)..][..dqkv]);
        }
        let mut v_i = Tensor::<f32>::default(&vec![total_seq_len, dqkv]);
        for x in 0..total_seq_len {
            let dst_v_i = &mut unsafe { v_i.data_mut() }[x * dqkv..][..dqkv];
            dst_v_i.copy_from_slice(&v.data()[((x * n_kv_h + i) * dqkv)..][..dqkv]);
        }
        for j in 0..n_groups {
            let mut q_ij = Tensor::<f32>::default(&vec![seq_len, dqkv]);
            for x in 0..seq_len {
                let dst_q_ij = &mut unsafe { q_ij.data_mut() }[x * dqkv..][..dqkv];
                dst_q_ij.copy_from_slice(&q.data()[(((x * n_kv_h + i) * n_groups + j) * dqkv)..][..dqkv]);
            }
            let score_offset = (i * n_groups + j) * seq_len * total_seq_len;
            let mut att_scores_ij = Tensor::<f32>::default(&vec![seq_len, total_seq_len]);
            OP::matmul_transb(&mut att_scores_ij, 0., &q_ij, &k_i, 1./(dqkv as f32).sqrt());
            (&mut score[score_offset..][..seq_len * total_seq_len]).copy_from_slice(att_scores_ij.data());
            let mut attn = att_scores_ij;
            OP::masked_softmax(&mut attn);
            let mut attn_v = Tensor::<f32>::default(&vec![seq_len, dqkv]);
            OP::matmul(&mut attn_v, 0., &attn, &v_i, 1.0);
            for x in 0..seq_len {
                let offset = ((x * n_kv_h + i) * n_groups + j) * dqkv;
                (0..dqkv).for_each(|i| hidden[offset + i] = attn_v.data()[x * dqkv + i]);
            }
        }
    }
}

// ``` python
// hidden = rms_norm(residual)
// gate = hidden @ gate_weight.T
// up = hidden @ up_weight.T
// act = gate * sigmoid(gate) * up ## SwiGLU
// output = act @ down_weight.T
// residual = output + residual
// ```
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // check shape
    let shape = residual.shape();
    assert!(shape.len() == 2 && gate.shape().len() == 2 && up.shape().len() == 2 && hidden_states.shape().len() == 2);
    let seq_len = shape[0];
    assert!(gate.shape()[0] == seq_len && up.shape()[0] == seq_len && shape == hidden_states.shape());
    let d = shape[1];
    let di = gate.shape()[1];
    assert!(w_gate.shape() == &vec![di, d] && w_up.shape() == &vec![di, d] && w_down.shape() == &vec![d, di]);
    // do calculation
    OP::rms_norm(hidden_states, &residual, rms_w, eps);
    OP::matmul_transb(gate, 0., &hidden_states, w_gate, 1.0);
    OP::matmul_transb(up, 0., &hidden_states, w_up, 1.0);
    let mut act = Tensor::<f32>::new(up.data().to_vec(), up.shape());
    OP::swiglu(&mut act, gate);
    OP::matmul_transb(residual, 1., &act, w_down, 1.);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
