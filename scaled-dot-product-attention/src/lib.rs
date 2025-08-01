use {
    candle_core::{
        D, Result, Tensor,
        scalar::{TensorOrScalar, TensorScalar},
    },
    candle_nn::ops::softmax_last_dim,
};

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    softmax_last_dim(&attn_weights)?.matmul(v)
}

pub trait TensorAllClose {
    fn all_close<T: TensorOrScalar>(&self, rhs: T, tolerance: f64) -> Result<bool>;
}

impl TensorAllClose for Tensor {
    fn all_close<T: TensorOrScalar>(&self, rhs: T, tolerance: f64) -> Result<bool> {
        let rhs = match rhs.to_tensor_scalar()? {
            TensorScalar::Tensor(rhs) => rhs,
            TensorScalar::Scalar(rhs) => rhs
                .to_dtype(self.dtype())?
                .to_device(self.device())?
                .broadcast_as(self.shape())?,
        };
        let shape = self.same_shape_binary_op(&rhs, "all_close")?;
        let all = self
            .sub(&rhs)?
            .abs()?
            .le(tolerance)?
            .to_dtype(DType::U32)?
            .sum_all()?;
        Ok(all.to_scalar::<u32>()? == shape.elem_count() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    use candle_core::{D, Result, Tensor};

    pub fn all_close(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> Result<bool> {
        let diff = (a.sub(b).unwrap()).abs()?;
        let tol = (b.abs()? * rtol)? + atol;
        let close = diff.le(&tol)?;
        Ok(close.all()?.to_scalar::<bool>()?)
    }

    #[test]
    fn test_basic_attention() -> Result<()> {
        let device = &Device::Cpu;
        // Simple case with 2 tokens and 3 dimensions
        let q = Tensor::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let k = Tensor::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let v = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], device)?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        // Expected output should be the same as v since q and k are identity-like
        let expected = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], device)?;
        assert!(output.close(&expected, 1e-5, 1e-5)?);
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
        assert!(output.all_close(&expected, 1e-5, 1e-5)?);
        Ok(())
    }

    #[test]
    fn test_softmax_effect() -> Result<()> {
        let device = &Device::Cpu;
        // Test that softmax is properly applied
        let q = Tensor::new(&[[1.0, 0.0]], device)?;
        let k = Tensor::new(&[[1.0, 0.0], [1.0, 1.0]], device)?;
        let v = Tensor::new(&[[1.0, 0.0], [0.0, 1.0]], device)?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        // First row of k matches perfectly with q, second row partially
        // Output should be weighted combination of v rows
        let expected = Tensor::new(&[[0.6225, 0.3775]], device)?;
        assert!(output.all_close(&expected, 1e-4, 1e-4)?);
        Ok(())
    }

    #[test]
    fn test_batch_processing() -> Result<()> {
        let device = &Device::Cpu;
        // Test with batch dimension
        let q = Tensor::new(
            &[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            device,
        )?;
        let k = Tensor::new(
            &[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            device,
        )?;
        let v = Tensor::new(
            &[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            device,
        )?;

        let output = scaled_dot_product_attention(&q, &k, &v)?;

        // First batch should be identity-like, second batch should be flipped
        let expected = Tensor::new(
            &[[[1.0, 2.0], [3.0, 4.0]], [[7.0, 8.0], [5.0, 6.0]]],
            device,
        )?;
        assert!(output.all_close(&expected, 1e-5, 1e-5)?);
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
