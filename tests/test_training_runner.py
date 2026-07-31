from training.deriv_client import TIMEFRAMES, get_all_instruments, get_all_timeframes
from training.training_runner import audit_training_scope, _filter_instruments


def test_training_scope_includes_all_requested_timeframes():
    instruments = get_all_instruments()
    timeframes = get_all_timeframes()
    audit = audit_training_scope(instruments, timeframes)
    assert audit["instrument_count"] == 44
    assert audit["timeframes"] == ["1m", "2m", "5m", "15m", "30m", "1h", "4h", "8h", "24h"]
    assert set(timeframes) == set(TIMEFRAMES)
    assert audit["combo_count"] == len(instruments) * len(timeframes)
    assert "forex" in audit["markets"]
    assert "synthetic_index" in audit["markets"]


def test_training_filter_supports_symbol_market_and_submarket():
    assert [i["symbol"] for i in _filter_instruments(symbols=["frxEURUSD"])] == ["frxEURUSD"]
    volatility = _filter_instruments(markets=["volatility"])
    assert volatility
    assert {i["submarket"] for i in volatility} == {"volatility"}
