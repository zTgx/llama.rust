/// superparameters

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub dim: i32,                // 模型维度
    pub n_layers: i32,           // Transformer的层数
    pub n_heads: i32,            // 注意⼒机制的头数
    pub n_kv_heads: i32,         // 键值头的数量
    pub vocab_size: i32,         // 词汇表⼤⼩
    pub hidden_dim: Option<i32>, // 隐藏层维度
    pub multiple_of: i32,
    pub norm_eps: f64,    // 归⼀化层的eps
    pub max_seq_len: i32, // 最⼤序列⻓度
    pub dropout: f64,     // dropout概率
    pub flash_attn: bool, // 是否使⽤Flash Attention
}

impl ModelConfig {
    pub fn new(
        dim: i32,
        n_layers: i32,
        n_heads: i32,
        n_kv_heads: i32,
        vocab_size: i32,
        hidden_dim: Option<i32>,
        multiple_of: i32,
        norm_eps: f64,
        max_seq_len: i32,
        dropout: f64,
        flash_attn: bool,
    ) -> Self {
        ModelConfig {
            model_type: "Tiny-K".to_string(),
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            hidden_dim,
            multiple_of,
            norm_eps,
            max_seq_len,
            dropout,
            flash_attn,
        }
    }

    // 计算模型总参数量
    pub fn total_params(&self) -> usize {
        let hidden_dim = self.hidden_dim.unwrap_or(self.dim * 4) as usize;
        (self.dim as usize * self.n_layers as usize * hidden_dim * 3)  // 注意力参数
        + (self.dim as usize * self.vocab_size as usize) // 词嵌入参数
    }
}
