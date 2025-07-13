use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::{EncodeInput, Result, Tokenizer};

fn main() -> Result<()> {
    let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
    let bpe = bpe_builder.dropout(0.1).unk_token("[UNK]".into()).build()?;

    let mut tokenizer = Tokenizer::new(bpe);

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}
