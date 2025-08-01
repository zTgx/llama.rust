use candle_core::{
    DType, Error, Result, Shape, Tensor,
    scalar::{TensorOrScalar, TensorScalar},
};

// https://github.com/huggingface/candle/pull/1549/files
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
        let shape = same_shape_binary_op(&self, &rhs, "all_close")?;
        let all = self
            .sub(&rhs)?
            .abs()?
            .le(tolerance)?
            .to_dtype(DType::U32)?
            .sum_all()?;
        Ok(all.to_scalar::<u32>()? == shape.elem_count() as u32)
    }
}

pub(crate) fn same_shape_binary_op<'a>(
    lhs: &'a Tensor,
    rhs: &Tensor,
    op: &'static str,
) -> Result<&'a Shape> {
    let lhs = lhs.shape();
    let rhs = rhs.shape();
    if lhs != rhs {
        Err(Error::ShapeMismatchBinaryOp {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            op,
        }
        .bt())
    } else {
        Ok(lhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, test_device};

    fn all_close(device: &Device) -> Result<()> {
        let t1 = Tensor::new(&[1.0004_f32, 1.0005], device)?;
        let t2 = Tensor::new(&[1.0005_f32, 1.0004], device)?;
        let x = t1.all_close(&t2, 0.001)?;
        let y = t1.all_close(&t2, 0.00001)?;
        assert_eq!(x, true);
        assert_eq!(y, false);
        Ok(())
    }

    #[test]
    fn all_close_works() {
        let device = Device::Cpu;

        all_close(&device).unwrap();
    }
}
