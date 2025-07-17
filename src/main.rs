use crate::model::InferenceEngine;
use crate::tokenizer::Tokenizer;
use crate::utils::Config;
use clap::Parser;

#[derive(Parser)]
#[command(name = "llama_inference")]
#[command(about = "A simple LLaMA inference engine using Candle")]
struct Args {
    #[arg(long, default_value = "Hello, world!")]
    prompt: String,
    #[arg(long, default_value_t = 20)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, short = 'm', required = true)]
    model: String, // 新增：必须指定模型 ID
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 加载配置
    let config = Config {
        model_id: args.model,
        max_seq_len: 2048,
        max_gen_tokens: args.max_tokens,
        temperature: args.temperature,
    };

    // 加载分词器
    let tokenizer = Tokenizer::new(&config.model_id)?;

    // 加载模型
    let mut engine = InferenceEngine::new(&config)?;

    // 执行推理
    let output = engine.infer(&args.prompt, &tokenizer, &config)?;
    println!("Output: {}", output);

    Ok(())
}
