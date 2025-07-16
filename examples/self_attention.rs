// Embeddings
// compute attention scores
// attetion weight: apply softmax to get attention weights
// context vector: compute weighted sum of values using attention weights

use candle_nn::{Embedding, Linear, Module, RmsNorm};
