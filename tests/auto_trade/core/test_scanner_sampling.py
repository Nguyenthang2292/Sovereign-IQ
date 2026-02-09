"""Test scanner sampling integration with Gemini Chart Scanner."""

import pytest

from modules.auto_trade.core.scanner_sampling import SamplingStrategy, sample_symbols


def test_random_sampling():
    """Test random sampling strategy."""
    symbols = [f"BTC/USDT:{i}" for i in range(100)]
    sampled = sample_symbols(symbols, sample_percentage=20, strategy="random")

    assert len(sampled) == 20
    assert all(s in symbols for s in sampled)


def test_stratified_sampling():
    """Test stratified sampling strategy."""
    symbols = [f"BTC/USDT:{i}" for i in range(100)]
    sampled = sample_symbols(symbols, sample_percentage=30, strategy="stratified")

    # Should sample approximately 30% (may vary slightly due to stratification)
    assert 25 <= len(sampled) <= 35
    assert all(s in symbols for s in sampled)


def test_volume_weighted_without_data_fetcher():
    """Test volume weighted falls back to random when no data_fetcher."""
    symbols = [f"BTC/USDT:{i}" for i in range(100)]
    # Without data_fetcher, should fall back to random sampling
    sampled = sample_symbols(symbols, sample_percentage=15, strategy="volume_weighted")

    assert len(sampled) == 15
    assert all(s in symbols for s in sampled)


def test_invalid_percentage_returns_all():
    """Test that invalid percentages return all symbols."""
    symbols = [f"BTC/USDT:{i}" for i in range(10)]

    # 0% should return all
    sampled_0 = sample_symbols(symbols, sample_percentage=0, strategy="random")
    assert sampled_0 == symbols

    # 100% should return all
    sampled_100 = sample_symbols(symbols, sample_percentage=100, strategy="random")
    assert sampled_100 == symbols

    # Negative should return all
    sampled_neg = sample_symbols(symbols, sample_percentage=-10, strategy="random")
    assert sampled_neg == symbols


def test_sampling_strategy_enum():
    """Test that SamplingStrategy enum values are accessible."""
    assert hasattr(SamplingStrategy, "RANDOM")
    assert hasattr(SamplingStrategy, "VOLUME_WEIGHTED")
    assert hasattr(SamplingStrategy, "STRATIFIED")
    assert hasattr(SamplingStrategy, "TOP_N_HYBRID")
    assert hasattr(SamplingStrategy, "SYSTEMATIC")
    assert hasattr(SamplingStrategy, "LIQUIDITY_WEIGHTED")

    assert SamplingStrategy.RANDOM.value == "random"
    assert SamplingStrategy.STRATIFIED.value == "stratified"


def test_top_n_hybrid_strategy():
    """Test top_n_hybrid sampling strategy."""
    symbols = [f"BTC/USDT:{i}" for i in range(100)]
    sampled = sample_symbols(symbols, sample_percentage=25, strategy="top_n_hybrid", top_percentage=50.0)

    # Should sample approximately 25%
    assert 20 <= len(sampled) <= 30
    assert all(s in symbols for s in sampled)


def test_systematic_strategy():
    """Test systematic sampling strategy."""
    symbols = [f"BTC/USDT:{i}" for i in range(100)]
    sampled = sample_symbols(symbols, sample_percentage=10, strategy="systematic")

    assert len(sampled) == 10
    assert all(s in symbols for s in sampled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
