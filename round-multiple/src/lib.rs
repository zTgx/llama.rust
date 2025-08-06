#[macro_export]
macro_rules! round_multiple_checked {
    ($x:expr, $m:expr) => {{
        let m = $m;
        assert!(m > 0, "m must be positive");
        (($x + m - 1) / m) * m
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_multiple_checked_works() {
        assert_eq!(round_multiple_checked!(7, 4), 8); // 7 → 8 (4×2)
        assert_eq!(round_multiple_checked!(10, 5), 10); // 10 → 10 (5×2)
        assert_eq!(round_multiple_checked!(11, 3), 12); // 11 → 12 (3×4)
    }

    #[test]
    #[should_panic(expected = "m must be positive")]
    fn round_multiple_checked_panic_with_zero_works() {
        round_multiple_checked!(10, 0);
    }
}
