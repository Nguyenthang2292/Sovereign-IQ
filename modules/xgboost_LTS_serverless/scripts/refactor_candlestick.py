import re

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/features/candlestick.rs", "r"
) as f:
    code = f.read()

# 1. Change struct fields from Vec<bool> to bool
code = re.sub(r"pub ([a-z_]+): Vec<bool>", r"pub \1: bool", code)

# 2. Change detect signature
code = code.replace(
    "pub fn detect(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Self",
    "pub fn detect_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> Self",
)

# 3. Change detect calls inside detect_at
code = re.sub(
    r"let ([a-z_]+) = detect_\1\(open, high, low, close\);", r"let \1 = detect_\1_at(open, high, low, close, i);", code
)

# 4. Change to_feature_vec signature
code = code.replace("pub fn to_feature_vec(&self, i: usize) -> Vec<f64>", "pub fn to_feature_vec(&self) -> Vec<f64>")

# 5. Change to_feature_vec body
# from: if self.doji.get(i).copied().unwrap_or(false) {
# to: if self.doji {
code = re.sub(r"if self\.([a-z_]+)\.get\(i\)\.copied\(\)\.unwrap_or\(false\) {", r"if self.\1 {", code)


# 6. Change all detect functions
def transform_detect_function(match):
    name = match.group(1)
    params = match.group(2)
    body = match.group(3)

    # new signature
    new_sig = f"fn detect_{name}_at({params}, i: usize) -> bool"

    # find the for loop start: "for i in K..open.len() {"
    # or "for i in K..close.len() {"
    m_loop = re.search(r"for i in (\d+)\.\.[a-z]+\.len\(\) \{", body)
    if not m_loop:
        # maybe it's just doing something else?
        return f"{new_sig} {{{body}}}"

    min_idx = int(m_loop.group(1))

    # Extract the inside of the loop
    # Find the corresponding '}' for the loop
    loop_start_idx = m_loop.end()
    balance = 1
    loop_end_idx = -1
    for idx in range(loop_start_idx, len(body)):
        if body[idx] == "{":
            balance += 1
        elif body[idx] == "}":
            balance -= 1
            if balance == 0:
                loop_end_idx = idx
                break

    inner_loop = body[loop_start_idx:loop_end_idx].strip()

    # replace result[i] = true with return true (and default return false)
    inner_loop = inner_loop.replace("result[i] = true;", "return true;")

    new_body = f"""
    if i < {min_idx} {{
        return false;
    }}
    {inner_loop}
    false
"""
    return f"{new_sig} {{{new_body}}}"


# We need to parse functions accurately. Regex might be tricky for nested braces.
code = re.sub(
    r"fn detect_([a-z_]+)\((.*?)\) -> Vec<bool> \{(.*?)\n\}", transform_detect_function, code, flags=re.DOTALL
)

with open(
    "c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/xgboost_LTS_serverless/src/features/candlestick.rs", "w"
) as f:
    f.write(code)

print("Done")
