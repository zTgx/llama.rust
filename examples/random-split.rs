use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::path::Path;

fn random_split(
    df: &DataFrame,
    train_frac: f64,
    validation_frac: f64,
) -> Result<(DataFrame, DataFrame, DataFrame)> {
    // 1. 获取所有行索引并打乱顺序
    let mut indices: Vec<usize> = (0..df.height()).collect();
    indices.shuffle(&mut thread_rng());

    // 2. 计算分割点
    let train_end = (df.height() as f64 * train_frac) as usize;
    let validation_end = train_end + (df.height() as f64 * validation_frac) as usize;

    // 3. 分割索引
    let train_indices = &indices[..train_end];
    let val_indices = &indices[train_end..validation_end];
    let test_indices = &indices[validation_end..];

    // 4. 根据索引提取数据
    let train_df = df.take(train_indices)?;
    let validation_df = df.take(val_indices)?;
    let test_df = df.take(test_indices)?;

    Ok((train_df, validation_df, test_df))
}

fn save_to_csv(df: &DataFrame, path: &str) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file).has_header(true).finish(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DataFrame {
        df! {
            "id" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        .unwrap()
    }

    #[test]
    fn test_split_proportions() -> Result<()> {
        let df = create_test_data();
        let (train, val, test) = random_split(&df, 0.7, 0.1)?;

        assert_eq!(train.height(), 7); // 10 * 0.7 = 7
        assert_eq!(val.height(), 1); // 10 * 0.1 = 1
        assert_eq!(test.height(), 2); // 剩余部分

        Ok(())
    }

    #[test]
    fn test_all_data_preserved() -> Result<()> {
        let df = create_test_data();
        let (train, val, test) = random_split(&df, 0.5, 0.3)?;

        // 合并后应包含所有原始行
        let combined = concat(&[train, val, test], UnionArgs::default())?;
        assert_eq!(combined.height(), df.height());

        Ok(())
    }
}

fn main() -> Result<()> {
    // 示例用法
    let df = df! {
        "id" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "text" => ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "label" => [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }?;

    // 随机分割 (70% 训练, 10% 验证, 20% 测试)
    let (train_df, val_df, test_df) = random_split(&df, 0.7, 0.1)?;

    // 保存为CSV
    save_to_csv(&train_df, "train.csv")?;
    save_to_csv(&val_df, "validation.csv")?;
    save_to_csv(&test_df, "test.csv")?;

    println!("数据集已保存为: train.csv, validation.csv, test.csv");
    Ok(())
}
