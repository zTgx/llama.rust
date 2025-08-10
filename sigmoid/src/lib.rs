use {
    candle_core::{Tensor, Result}, candle_nn::Module,
};

pub struct Sigmoid {
    pub params: Vec<Tensor>,
    pub grads: Vec<Tensor>,
    pub out: Option<Tensor>,
}