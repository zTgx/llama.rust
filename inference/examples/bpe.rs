use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
    let bpe = bpe_builder.dropout(0.1).unk_token("[UNK]".into()).build()?;

    let tokenizer = Tokenizer::new(bpe);

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}

// from transformers import GPT2Tokenizer

// tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
// text = "The cat sat on the mat."

// # 分词并添加特殊token
// inputs = tokenizer(text, return_tensors="pt")

// # 自动生成输入-目标对（GPT类模型）
// input_ids = inputs.input_ids[:, :-1]  # 输入：去掉最后一个token
// labels = inputs.input_ids[:, 1:]      # 目标：去掉第一个token
