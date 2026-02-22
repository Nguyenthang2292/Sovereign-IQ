import os

replacements = [
    # property_tests.rs
    (
        'ma_type in prop_oneof![\n            Just("EMA".to_string()),\n            Just("HMA".to_string()),\n            Just("WMA".to_string()),\n            Just("DEMA".to_string()),\n            Just("LSMA".to_string()),\n            Just("KAMA".to_string()),\n        ]',
        "ma_type in prop_oneof![\n            Just(atc_serverless::MAType::Ema),\n            Just(atc_serverless::MAType::Hma),\n            Just(atc_serverless::MAType::Wma),\n            Just(atc_serverless::MAType::Dema),\n            Just(atc_serverless::MAType::Lsma),\n            Just(atc_serverless::MAType::Kama),\n        ]",
    ),
    ("match ma_type.as_str() {", "match ma_type {"),
    ('"EMA" =>', "atc_serverless::MAType::Ema =>"),
    ('"HMA" =>', "atc_serverless::MAType::Hma =>"),
    ('"WMA" =>', "atc_serverless::MAType::Wma =>"),
    ('"DEMA" =>', "atc_serverless::MAType::Dema =>"),
    ('"LSMA" =>', "atc_serverless::MAType::Lsma =>"),
    ('"KAMA" =>', "atc_serverless::MAType::Kama =>"),
    ('ma_type: "EMA".to_string()', "ma_type: atc_serverless::MAType::Ema"),
    ('ma_type: "HMA".to_string()', "ma_type: atc_serverless::MAType::Hma"),
    ('ma_type: "WMA".to_string()', "ma_type: atc_serverless::MAType::Wma"),
    ('ma_type: "DEMA".to_string()', "ma_type: atc_serverless::MAType::Dema"),
    ('ma_type: "LSMA".to_string()', "ma_type: atc_serverless::MAType::Lsma"),
    ('ma_type: "KAMA".to_string()', "ma_type: atc_serverless::MAType::Kama"),
    (
        'let ma_types = vec!["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"];',
        "let ma_types = vec![atc_serverless::MAType::Ema, atc_serverless::MAType::Hma, atc_serverless::MAType::Wma, atc_serverless::MAType::Dema, atc_serverless::MAType::Lsma, atc_serverless::MAType::Kama];",
    ),
    ("ma_type: ma_type.to_string(),", "ma_type: ma_type.clone(),"),
    ('prices_arr.view(),\n        "EMA",', "prices_arr.view(),\n        &atc_serverless::MAType::Ema,"),
    (
        'calculate_layer1_signal(\n        prices_arr.view(),\n        "EMA",',
        "calculate_layer1_signal(\n        prices_arr.view(),\n        &atc_serverless::MAType::Ema,",
    ),
]

for root, dirs, files in os.walk("tests"):
    for file in files:
        if file.endswith(".rs"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content
            for k, v in replacements:
                content = content.replace(k, v)
            if content != original_content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
