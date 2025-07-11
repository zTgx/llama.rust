use anyhow::Result;
use polars::prelude::*;

fn encode_labels(df: &mut DataFrame) -> Result<()> {
    // 将 "ham"/"spam" 替换为 0/1
    let label_series = df.column("Label")?.utf8()?;
    let encoded = label_series
        .into_iter()
        .map(|opt_label| match opt_label {
            Some("ham") => Some(0),
            Some("spam") => Some(1),
            _ => None, // 处理其他意外值
        })
        .collect::<UInt32Chunked>(); // 存储为无符号整数

    // 替换原列
    df.replace("Label", encoded.into_series())?;
    Ok(())
}

// 使用示例
fn main() -> Result<()> {
    let mut df = df! {
        "Text" => ["hello", "spam offer", "normal msg"],
        "Label" => ["ham", "spam", "ham"]
    }?;

    encode_labels(&mut df)?;
    println!("{:?}", df);
    Ok(())
}
