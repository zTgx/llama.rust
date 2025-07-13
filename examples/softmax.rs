use candle_core::{Device, Tensor, test_utils::to_vec2_round};

fn main() {
    let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu).unwrap();
    let a = candle_nn::ops::softmax(&a, 1).unwrap();
    assert_eq!(
        to_vec2_round(&a, 4).unwrap(),
        &[
            [0.1345, 0.3655, 0.1345, 0.3655],
            [0.0049, 0.2671, 0.7262, 0.0018]
        ]
    );
}

// Attention score -> softmax -> attention weights

// The softmax function is used to normalize the attention scores so that they sum to 1
// and can be interpreted as probabilities. This is crucial for the attention mechanism to work correctly,
// as it allows the model to focus on different parts of the input sequence based on their relevance
// to the current token being processed.