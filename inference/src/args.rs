use clap::Parser;

/// LLaMA 推理引擎配置参数
#[derive(Parser, Debug)]
#[command(name = "llama-serve")]
#[command(
    about = "A simple LLaMA inference engine using Candle",
    version,
    author
)]
pub struct Args {
    /// 模型名称或本地检查点路径（Hugging Face格式，如 `meta-llama/Llama-3-70B`）
    #[arg(short, long)]
    pub model: String,

    /// 输入提示文本（需用引号包裹）
    #[arg(short, long)]
    pub prompt: String,

    /// Device: CPU or CUDA
    #[arg(long)]
    pub cpu: bool,

    /// 生成温度（0.0-1.0，值越高随机性越强）
    #[arg(short, long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// 最大生成token数量
    #[arg(short = 'n', long, default_value_t = 100)]
    pub max_tokens: usize,

    /// 是否启用调试模式（打印详细日志）
    #[arg(short, long, default_value_t = false)]
    pub debug: bool,
}
