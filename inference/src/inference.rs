/*
 * Adapted from
 * https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama2-c/main.rs
 * Copyright (c) 2023, The Huggingface team.
 */

use {
    crate::{args::Args, tokenizer::Tokenizer},
    candle_core::{D, Tensor},
};

use candle_transformers::models::llama2_c as model;
// use candle_transformers::models::llama2_c_weights as weights;
// use candle_transformers::models::quantized_llama2_c as qmodel;
use anyhow::{Error as E, Result};
// use clap::builder::Str;
use model::{Cache, Config as ModelConfig};
// use qmodel::QLlama;
use candle_core::DType;
use candle_core::safetensors;
use candle_transformers::generation::LogitsProcessor;

use candle_core::IndexOp;
use std::io::Write;

enum Model {
    Llama(model::Llama),
    // QLlama(QLlama),
}

impl Model {
    fn forward(&self, xs: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        match self {
            Self::Llama(l) => Ok(l.forward(xs, pos, cache)?),
            // Self::QLlama(l) => Ok(l.forward(xs, pos, cache)?),
        }
    }
}

pub struct InferenceEngine;
impl InferenceEngine {
    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        args: &Args,
    ) -> anyhow::Result<(f64, Vec<String>)> {
        let mut rests = Vec::<String>::new();

        let device = crate::device(args.cpu)?;

        // Load model: safetensor
        let config = ModelConfig::tiny_15m();
        let tensors = safetensors::load(args.model.clone(), &device)?;
        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &device);
        let mut cache = model::Cache::new(true, &config, vb.pp("rot"))?;
        let model = Model::Llama(model::Llama::load(vb, config.clone())?);

        println!("starting the inference loop");
        let mut logits_processor =
            LogitsProcessor::new(299792458, Some(args.temperature), args.top_p);
        let mut index_pos = 0;

        print!("{}", prompt);
        let mut tokens = tokenizer.encode(prompt).map_err(E::msg)?;
        let mut tokenizer =
            crate::token_output_stream::TokenOutputStream::new(tokenizer.clone().tokenizer);

        let start_gen = std::time::Instant::now();
        for index in 0.. {
            if tokens.len() >= config.seq_len {
                break;
            }
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, index_pos, &mut cache)?;
            let logits = logits.i((0, logits.dim(1)? - 1))?;
            let logits = if args.repeat_penalty == 1. || tokens.is_empty() {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
            rests.push(rest);
        }

        let dt = start_gen.elapsed();
        println!(
            "\n{} tokens generated ({:.2} token/s)\n",
            tokens.len(),
            tokens.len() as f64 / dt.as_secs_f64(),
        );

        Ok((dt.as_secs_f64(), rests))
    }

    pub fn sample_token(logits: &Tensor, _temperature: f32) -> anyhow::Result<u32> {
        // Simple top-1 sampling for minimal implementation
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let token = probs.argmax(D::Minus1)?.to_scalar::<u32>()?;
        Ok(token)
    }
}
