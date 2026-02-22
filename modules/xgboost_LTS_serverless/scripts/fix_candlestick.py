import os

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/features/candlestick.rs", "r"
) as f:
    text = f.read()

# Fix `pub fn detect_at_at`
text = text.replace(
    "pub fn detect_at_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> Self",
    "pub fn detect(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> Self",
)

# Fix dangling parameters like:
# close: &[f64],
# , i: usize) -> bool {
text = text.replace(",\n, i: usize) -> bool {", ",\n    i: usize,\n) -> bool {")

# Also `i` is not always passed with 6 arguments in detect_at.. Wait, the closures took `|i| {}`
# So I should check what my regex did to `let three_white_soldiers = ...`
# The original code `let three_white_soldiers = detect_three_white_soldiers(open, high, low, close);`
# Was replaced to `detect_three_white_soldiers_at(open, high, low, close, i)`
# Let me fix the remaining `detect_xyz_at(..., i)` where the function definition takes 6 inputs... wait, the function `detect_three_white_soldiers_at` takes 5 inputs: `open, high, low, close, i`.
# Why did rustc say `error[E0061]: this function takes 6 arguments but 5 arguments were supplied`?
# Let's look at the function signature:
# fn detect_three_white_soldiers_at(
#     open: &[f64],
#     high: &[f64],
#     _low: &[f64],
#     close: &[f64],
# , i: usize) -> bool {
# Wait, `_low: &[f64],`
# `close: &[f64],`
# `, i: usize) -> bool {`
# That's 5 arguments! Why did Rust complain `argument #6 of type usize is missing`??
# Oh! Because the trailing comma means `, i: usize` is the 6th token in the argument list?
# No, `open, high, _low, close` + `, i: usize` -> 5 arguments!
# Let me look closely:
# fn detect_three_white_soldiers_at(
#     open: &[f64],
#     high: &[f64],
#     _low: &[f64],
#     close: &[f64],
#     i: usize
# ) -> bool {
# Let's fix the `, i: usize) -> bool {` first.
import re

text = re.sub(r",\n\s*, i: usize\) -> bool \{", r",\n    i: usize\n) -> bool {", text)

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/features/candlestick.rs", "w"
) as f:
    f.write(text)

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/feature_engine.rs", "r"
) as f:
    engine_text = f.read()

# Fix the call in feature_engine.rs!
engine_text = engine_text.replace(
    "let patterns = features::candlestick::CandlestickPatterns::detect(\n            &data.open,\n            &data.high,\n            &data.low,\n            &data.close,\n        );",
    "let patterns = features::candlestick::CandlestickPatterns::detect(\n            &data.open,\n            &data.high,\n            &data.low,\n            &data.close,\n            i,\n        );",
)

# one-line variant just in case
engine_text = engine_text.replace(
    "CandlestickPatterns::detect_at(&data.open, &data.high, &data.low, &data.close)",
    "CandlestickPatterns::detect(&data.open, &data.high, &data.low, &data.close, i)",
)
engine_text = engine_text.replace(
    "CandlestickPatterns::detect(&data.open, &data.high, &data.low, &data.close)",
    "CandlestickPatterns::detect(&data.open, &data.high, &data.low, &data.close, i)",
)

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/feature_engine.rs", "w"
) as f:
    f.write(engine_text)

print("Fixed syntax")
