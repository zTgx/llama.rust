use {
    anyhow::Result,
    clap::Parser,
    llama_rust::{config::Config, model::InferenceEngine, tokenizer::Tokenizer},
};

/// Pretrain 分词模型
pub const PRETRAIN_TOKENIZER_BERT_BASE_CASED: &str = "bert-base-cased";
pub const PRETRAIN_TOKENIZER_GPT2: &str = "gpt2";

/// LLaMA 推理引擎配置参数
#[derive(Parser, Debug)]
#[command(name = "llama_inference")]
#[command(
    about = "A simple LLaMA inference engine using Candle",
    version,
    author
)]
struct Args {
    /// 模型名称或本地检查点路径（Hugging Face格式，如 `meta-llama/Llama-3-70B`）
    #[arg(short, long)]
    model: String,

    /// 输入提示文本（需用引号包裹）
    #[arg(short, long)]
    prompt: String,

    /// 生成温度（0.0-1.0，值越高随机性越强）
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// 最大生成token数量
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// 是否启用调试模式（打印详细日志）
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化配置
    let config = Config {
        model_id: args.model.clone(),
        max_seq_len: 2048,
        max_gen_tokens: args.max_tokens,
        temperature: args.temperature,
    };

    // 加载分词器
    let tokenizer = Tokenizer::new(PRETRAIN_TOKENIZER_BERT_BASE_CASED)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 加载模型（显式传递调试标志）
    let mut engine = InferenceEngine::new(&config)
        .map_err(|e| anyhow::anyhow!("Failed to initialize engine: {}", e))?;

    // 执行推理并处理输出
    let (gen_time, ret) = engine.generate(&args.prompt, &tokenizer, &config)?;

    // 输出
    println!(
        "\nachieved tok/s: {}",
        ((ret.len() - 1) as f32 / gen_time as f32) * 1000.0
    );

    Ok(())
}
