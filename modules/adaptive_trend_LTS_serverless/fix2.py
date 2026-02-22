import os

for root, _, files in os.walk("tests"):
    for file in files:
        if file.endswith(".rs"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            content = content.replace('ma_type: "EMA".to_string(),', "ma_type: atc_serverless::MAType::Ema,")
            content = content.replace('ma_type: "HMA".to_string(),', "ma_type: atc_serverless::MAType::Hma,")
            content = content.replace('ma_type: "WMA".to_string(),', "ma_type: atc_serverless::MAType::Wma,")
            content = content.replace('ma_type: "DEMA".to_string(),', "ma_type: atc_serverless::MAType::Dema,")
            content = content.replace('ma_type: "LSMA".to_string(),', "ma_type: atc_serverless::MAType::Lsma,")
            content = content.replace('ma_type: "KAMA".to_string(),', "ma_type: atc_serverless::MAType::Kama,")

            content = content.replace('Just("EMA".to_string()),', "Just(atc_serverless::MAType::Ema),")
            content = content.replace('          Just("SMA".to_string()),\n', "")
            content = content.replace('            Just("SMA".to_string()),\n', "")
            content = content.replace('Just("WMA".to_string()),', "Just(atc_serverless::MAType::Wma),")
            content = content.replace('Just("DEMA".to_string()),', "Just(atc_serverless::MAType::Dema),")
            content = content.replace('Just("HMA".to_string())', "Just(atc_serverless::MAType::Hma)")

            content = content.replace(
                'prices_arr.view(),\n        "EMA",', "prices_arr.view(),\n        &atc_serverless::MAType::Ema,"
            )

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
