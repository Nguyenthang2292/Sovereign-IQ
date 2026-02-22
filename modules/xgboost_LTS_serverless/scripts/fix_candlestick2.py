import re

path = "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/features/candlestick.rs"
with open(path, "r") as f:
    text = f.read()


# Fix all function definitions starting with `fn detect_` but not ending with `_at`
# that are called from `detect` function block. Just check for fn detect_([a-z_]+)\(
def fix_func_name(match):
    name = match.group(1)
    if not name.endswith("_at"):
        return f"fn detect_{name}_at("
    return match.group(0)


text = re.sub(r"fn detect_([a-z_]+)\(", fix_func_name, text)

# fix the unused variables by renaming to _close
text = text.replace(
    "fn detect_rising_window_at(_open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {",
    "fn detect_rising_window_at(_open: &[f64], high: &[f64], low: &[f64], _close: &[f64], i: usize) -> bool {",
)
text = text.replace(
    "fn detect_falling_window_at(_open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {",
    "fn detect_falling_window_at(_open: &[f64], high: &[f64], low: &[f64], _close: &[f64], i: usize) -> bool {",
)


with open(path, "w") as f:
    f.write(text)

print("Fixed")
