use {
    anyhow::Result,
    clap::Parser,
    llama_rust::args::Args,
    llama_rust::{inference::InferenceEngine, tokenizer::Tokenizer},
};

/// Pretrain 分词模型
pub const PRETRAIN_TOKENIZER_BERT_BASE_CASED: &str = "bert-base-cased";
pub const PRETRAIN_TOKENIZER_GPT2: &str = "gpt2";

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{:?}", args);

    // 加载分词器
    let tokenizer = Tokenizer::new(PRETRAIN_TOKENIZER_BERT_BASE_CASED)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("loaded tokenizer.");

    // 执行推理并处理输出
    let (_gen_time, ret) = InferenceEngine.generate(&args.prompt, &tokenizer, &args)?;

    // 输出
    println!("Ret: {:?}", ret);

    Ok(())
}
