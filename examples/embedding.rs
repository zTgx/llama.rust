// embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
use candle_nn::{Embedding, Linear, Module, RmsNorm};

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}
