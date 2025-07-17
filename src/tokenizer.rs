use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    tokenizer: HFTokenizer,
    eos_token_id: u32,
}

impl Tokenizer {
    pub fn new(model_id: &str) -> anyhow::Result<Self> {
        let tokenizer = HFTokenizer::from_pretrained(model_id, None)
            .map_err(|_| anyhow::anyhow!("Failed to load tokenizer for model {}", model_id))?;

        let eos_token_id = 50247; // TODO:
        Ok(Self {
            tokenizer,
            eos_token_id,
        })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        let text = self.tokenizer.decode(tokens, true).unwrap();
        Ok(text)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}
