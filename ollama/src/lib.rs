use ollama_rs::{models::LocalModel, Ollama};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashSet;
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelList {
    pub models: Vec<Model>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub model: String,
    pub modified_at: DateTime<Utc>,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails {
    pub parent_model: Option<String>,
    pub format: String,
    pub family: String,
    pub families: HashSet<String>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

pub async fn tags(ollama: &Ollama) -> Result<LocalModel> {
    todo!("Implement the tags function to fetch local models from Ollama");
}