import pandas as pd

from backtesting.data_loader import load_excel_ohlcv


def test_load_excel_ohlcv_respects_synonyms_and_resamples(tmp_path):
    raw = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-01"],
            "Time": ["00:00", "00:01"],
            "Open Price": [1.0, 2.0],
            "High Price": [1.1, 2.2],
            "Low Price": [0.9, 1.9],
            "Close Price": [1.05, 2.05],
            "Tick Volume": [10, 20],
        }
    )
    workbook = tmp_path / "example.xlsx"
    raw.to_excel(workbook, index=False)

    cleaned = load_excel_ohlcv(workbook, resample_rule="2T")

    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert cleaned.index[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert cleaned.iloc[0].to_dict() == {
        "Open": 1.0,
        "High": 2.2,
        "Low": 0.9,
        "Close": 2.05,
        "Volume": 30,
    }
