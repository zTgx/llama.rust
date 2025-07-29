/// 模型配置
pub struct Config {
    pub model_id: String,
    pub max_seq_len: usize,
    pub max_gen_tokens: usize,
    pub temperature: f64,
}
