use crate::config::Config;
use crate::tokenizer::Tokenizer;
use candle_core::{D, Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use candle_transformers::models::llama::{Llama, LlamaConfig};

pub struct InferenceEngine {
    model: Llama,
    device: Device,
}

impl InferenceEngine {
    pub fn new(config: &Config) -> anyhow::Result<Self> {
        let device = Device::cuda_if_available(0)?;
        // 动态加载模型权重
        let model_path = format!("{}/model.safetensors", config.model_id);
        let vb = VarBuilder::from_safetensors(&[model_path], &device)?;

        // 使用默认配置，假设模型兼容 LLaMA 架构
        // 注意：实际生产中需根据 model_id 动态加载 config.json
        let llama_config = LlamaConfig::default();
        let model = Llama::load(vb, &llama_config)?; // TODO: 从hf-hub加载config.json并解析
        // let (hf_model_id, revision) = hf_hub::api::sync::Api::new()?
        //     .model_info(&config.model_id)?
        //     .config
        //     .and_then(|v| v.get("auto").cloned())
        //     .and_then(|v| v.as_object().cloned())
        //     .and_then(|v| v.get("model_id").cloned())
        //     .and_then(|v| v.as_str().map(|s| s.to_string()))
        //     .map(|model_id| (model_id, None))
        //     .unwrap_or_else(|| (config.model_id.clone(), None));
        // let config_filename = hf_hub::api::sync::Api::new()?
        //     .model(&hf_model_id)
        //     .get("config.json")?;
        // let llama_config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

        Ok(Self { model, device })
    }

    pub fn infer(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        config: &Config,
    ) -> anyhow::Result<String> {
        // Tokenize the input prompt
        let tokens = tokenizer.encode(prompt)?;
        let mut input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;

        // Initialize output tokens
        let mut output_tokens = tokens;

        // Generation loop
        for _ in 0..config.max_gen_tokens {
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;

            // Apply temperature sampling
            let next_token = Self::sample_token(&logits, config.temperature)?;

            // Append token to output
            output_tokens.push(next_token);

            // Prepare next input
            input = Tensor::new(&output_tokens, &self.device)?.unsqueeze(0)?;

            // Stop at EOS token
            if next_token == tokenizer.eos_token_id() {
                break;
            }
        }

        // Decode output tokens
        let output = tokenizer.decode(&output_tokens)?;
        Ok(output)
    }

    fn sample_token(logits: &Tensor, temperature: f32) -> anyhow::Result<u32> {
        // Simple top-1 sampling for minimal implementation
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let token = probs.argmax(D::Minus1)?.to_scalar::<u32>()?;
        Ok(token)
    }
}
pub fn config(&self) -> &LlamaConfig {
    self.model.config()
}
