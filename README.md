<div align="center">

# llama.rust   
LLM inference in Rust

</div>

## Inference
```bash
cargo run --release -p llama-serve -- --model "meta-llama/Llama-3.2-1B" 
    --prompt "What is the capital of France?" --max-tokens 20 --temperature 0.7
```

## Parameters
- `--model`: Path to the Hugging Face model ID.
- `--prompt`: The prompt to use for inference.
- `--max-tokens`: The maximum number of tokens to generate.
- `--temperature`: The temperature to use for sampling.

## References
This project draws inspiration from:
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llm](https://github.com/rustformers/llm)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [candle](https://github.com/huggingface/candle)