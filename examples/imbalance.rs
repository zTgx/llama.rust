use anyhow::Result;
use polars::prelude::*;
use rand::rng;

fn create_balanced_dataset(df: &DataFrame) -> Result<DataFrame> {
    // 1. 统计 spam 的数量
    let spam_count = df
        .clone()
        .filter(col("Label").eq(lit("spam")))
        .collect()?
        .height();

    // 2. 随机采样 ham 使其数量与 spam 相同
    let ham_df = df
        .lazy()
        .filter(col("Label").eq(lit("ham"))) // This line already uses lazy, so it's fine.
        .collect()?;

    let ham_indices: Vec<usize> = (0..ham_df.height()).collect();
    let mut rng = rng();
    let sampled_indices = ham_indices
        .choose_multiple(&mut rng, spam_count)
        .cloned()
        .collect::<Vec<_>>();

    let sampled_ham = ham_df.take(&sampled_indices)?;

    // 3. 合并 spam 和采样后的 ham
    let spam_df = df.clone().filter(col("Label").eq(lit("spam"))).collect()?;

    let balanced_df = concat(&[sampled_ham, spam_df], UnionArgs::default())?;

    Ok(balanced_df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    fn create_test_data() -> DataFrame {
        df! {
            "Text" => ["spam msg", "ham msg", "ham msg", "spam msg", "ham msg"],
            "Label" => ["spam", "ham", "ham", "spam", "ham"]
        }
        .unwrap()
    }

    #[test]
    fn test_balance_dataset() -> Result<()> {
        // 1. 准备测试数据 (3 ham, 2 spam)
        let df = create_test_data();

        // 2. 执行平衡
        let balanced_df = create_balanced_dataset(&df)?;

        // 3. 验证结果
        let counts = balanced_df.column("Label")?.value_counts(false, false)?;

        // 转换为 HashMap 方便验证
        let count_map: HashMap<_, _> = counts
            .column("Label")?
            .utf8()?
            .into_iter()
            .zip(counts.column("counts")?.u32()?.into_iter())
            .collect();

        // 断言 spam 和 ham 数量相同
        assert_eq!(count_map["spam"], count_map["ham"]);
        // 断言总数正确 (2 spam + 2 sampled ham)
        assert_eq!(balanced_df.height(), 4);

        Ok(())
    }

    #[test]
    fn test_empty_input() -> Result<()> {
        let empty_df = DataFrame::default();
        assert!(create_balanced_dataset(&empty_df).is_err());
        Ok(())
    }

    #[test]
    fn test_single_class() -> Result<()> {
        let df = df! {
            "Text" => ["msg1", "msg2"],
            "Label" => ["spam", "spam"]
        }?;
        assert!(create_balanced_dataset(&df).is_err());
        Ok(())
    }
}

fn main() -> Result<()> {
    // 示例用法
    let df = df! {
        "Text" => ["spam1", "ham1", "ham2", "spam2", "ham3", "ham4"],
        "Label" => ["spam", "ham", "ham", "spam", "ham", "ham"]
    }?;

    let balanced_df = create_balanced_dataset(&df)?;
    println!("Balanced Dataset:");
    println!("{:?}", balanced_df);

    Ok(())
}
