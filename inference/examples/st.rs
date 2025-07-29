use ndarray::Array;
use ndarray::s;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;

fn main() {
    // 常量定义
    const TRUE_B: f32 = 1.0;
    const TRUE_W: f32 = 2.0;
    const N: usize = 100;
    const SEED: u64 = 42;

    // 1. 生成随机数生成器（固定种子）
    // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    let mut rng = Isaac64Rng::seed_from_u64(SEED);

    // 2. 生成 x (N, 1) 的随机数组（范围 [0, 1)）
    let x = Array::random_using((N, 1), Uniform::new(0.0, 1.0), &mut rng);

    // 3. 生成 epsilon ~ N(0, 0.1^2) 的高斯噪声
    let epsilon = Array::random_using((N, 1), Uniform::new(0., 0.1), &mut rng);

    // 4. 计算 y = true_b + true_w * x + epsilon
    let y = TRUE_B + TRUE_W * &x + epsilon;

    // 打印部分结果验证
    println!("x (first 5):\n{:?}", x.slice(s![..5, ..]));
    println!("y (first 5):\n{:?}", y.slice(s![..5, ..]));
}
