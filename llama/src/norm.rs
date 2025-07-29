use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};

/// RMSNorm 实现，类似于 LayerNorm 但仅调整均方根
/// 公式：output = (x / sqrt(mean(x^2) + eps)) * weight
#[derive(Debug)]
pub struct RMSNorm {
    eps: f64,       // 防止除以0的小常数
    weight: Tensor, // 可学习的缩放参数（初始化为1）
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // 初始化权重参数，使用常量1初始化
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        Ok(Self { eps, weight })
    }

    /// 核心计算部分（对应Python的_norm方法）
    /// 计算步骤：
    /// 1. 计算输入x的平方均值（沿最后一个维度）
    /// 2. 取平方根的倒数（使用rsqrt优化性能）
    /// 3. 乘以原始输入x
    fn norm(&self, x: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = x.dims3()?;
        // 计算平方均值（保持维度以便广播）
        let mean_x2 = x.sqr()?.mean_keepdim(2)?; // 沿最后一个维度(hidden_dim)计算
        // 计算 RMS 分母（加上eps防止除零）
        let rms = (mean_x2 + self.eps)?.sqrt()?;
        // 应用归一化
        x * rms
    }
}

impl Module for RMSNorm {
    /// 前向传播
    /// 步骤：
    /// 1. 将输入转换为float32类型（确保计算精度）
    /// 2. 应用RMSNorm计算
    /// 3. 乘以可学习的权重参数
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 转换为f32类型计算（类似Python中的.float()）
        let x_f32 = x.to_dtype(DType::F32)?;
        // 应用RMSNorm
        let normalized = self.norm(&x_f32)?;
        // 应用权重并返回原数据类型
        let weighted = (&normalized * &self.weight)?;
        weighted.to_dtype(x.dtype())
    }
}

// 使用示例
fn example_usage() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 初始化RMSNorm层（隐藏层维度=768, eps=1e-5）
    let rms_norm = RMSNorm::new(768, 1e-5, vb)?;

    // 创建随机输入（batch_size=2, seq_len=10, hidden_dim=768）
    let x = Tensor::randn(0f32, 1.0, &[2, 10, 768], &device)?;

    // 前向计算
    let output = rms_norm.forward(&x)?;
    println!("Output shape: {:?}", output.shape());

    Ok(())
}
