# 📚 Documentation

Thư mục này chứa tất cả các file tài liệu (.md) của project để dễ quản lý và tách biệt khỏi code.

## Danh sách tài liệu

### Core Modules
- **[DataFetcher.md](./DataFetcher.md)** - Tài liệu về DataFetcher class, cách lấy dữ liệu OHLCV và giá hiện tại từ exchanges
- **[ExchangeManager.md](./ExchangeManager.md)** - Tài liệu về ExchangeManager, AuthenticatedExchangeManager và PublicExchangeManager
- **[PortfolioCorrelationAnalyzer.md](./PortfolioCorrelationAnalyzer.md)** - Tài liệu về PortfolioCorrelationAnalyzer, phân tích correlation giữa portfolio và symbols

### Deep Learning Modules
- **[feature_selection.md](./feature_selection.md)** - Tài liệu về FeatureSelector, chọn lọc và kỹ thuật hóa features cho deep learning
- **[deeplearning_data_pipeline.md](./deeplearning_data_pipeline.md)** - Tài liệu về DeepLearningDataPipeline, pipeline chuẩn bị data cho TFT
- **[deeplearning_dataset.md](./deeplearning_dataset.md)** - Tài liệu về TFTDataModule, tạo TimeSeriesDataSet và DataLoaders cho TFT

## Cấu trúc

```
docs/
├── README.md                      # File này
├── DataFetcher.md                 # Tài liệu DataFetcher
├── ExchangeManager.md              # Tài liệu ExchangeManager
├── PortfolioCorrelationAnalyzer.md # Tài liệu PortfolioCorrelationAnalyzer
├── feature_selection.md            # Tài liệu FeatureSelector
├── deeplearning_data_pipeline.md   # Tài liệu DeepLearningDataPipeline
└── deeplearning_dataset.md         # Tài liệu TFTDataModule
```

## Lưu ý

- Tất cả các file documentation (.md) nên được đặt trong thư mục này
- Các link nội bộ giữa các file .md sử dụng relative path (ví dụ: `./ExchangeManager.md`)
- Không nên đặt file .md trong thư mục `modules/` để tránh lẫn với code

