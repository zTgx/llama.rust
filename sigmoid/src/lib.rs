use {
    candle_core::{Tensor, Result}, candle_nn::{ops::sigmoid, Module},
};

pub struct Sigmoid {
    pub params: Vec<Tensor>,
    pub grads: Vec<Tensor>,
    pub out: Option<Tensor>,
}

impl Module for Sigmoid {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        sigmoid(xs)
    }
}
