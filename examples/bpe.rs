//OpenAI Byte Pair Encoders

use bpe_openai::cl100k;

fn main() {
    let bpe = cl100k();
    let count = bpe.count("Hello, world!");
    println!("{tokens}");
}
