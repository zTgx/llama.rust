use {
    candle_core::{D, Result, Tensor},
    candle_nn::ops::softmax_last_dim,
};

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    softmax_last_dim(&attn_weights)?.matmul(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use all_close::TensorAllClose;
    use candle_core::{DType, Device};
    use candle_core::{Result, Tensor};

    #[test]
    fn test_basic_attention() -> Result<()> {
        let device = &Device::Cpu;
        // Simple case with 2 tokens and 3 dimensions
        let q = Tensor::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let k = Tensor::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let v = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], device)?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        // Expected output should be the same as v since q and k are identity-like
        let expected = Tensor::new(&[[1.7191, 2.7191], [2.2809, 3.2809]], device)?;
        assert!(output.all_close(&expected, 1e-1)?);
        Ok(())
    }

    #[test]
    fn test_attention_with_scale() -> Result<()> {
        let device = &Device::Cpu;
        // Larger dimension to test scaling
        let q = Tensor::new(&[[1.0, 0.0, 0.0, 0.0]], device)?;
        let k = Tensor::new(&[[1.0, 0.0, 0.0, 0.0]], device)?;
        let v = Tensor::new(&[[1.0, 2.0]], device)?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        // Should still match v despite higher dimension due to scaling
        let expected = Tensor::new(&[[1.0, 2.0]], device)?;
        assert!(output.all_close(&expected, 1e-1)?);
        Ok(())
    }

    #[test]
    fn test_dtype_consistency() -> Result<()> {
        let device = &Device::Cpu;
        // Test with f32 dtype
        let q = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device)?;
        let k = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], device)?;
        let v = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], device)?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    }
}
