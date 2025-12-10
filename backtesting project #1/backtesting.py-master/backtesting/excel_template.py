from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import pandas as pd

EXCEL_SCHEMA: Mapping[str, object] = {
    "Trades": {
        "description": "One row per trade. Base table for all other summaries.",
        "columns": [
            {"name": "trade_id", "dtype": "str"},
            {"name": "pair", "dtype": "str"},
            {"name": "direction", "dtype": "str"},
            {"name": "entry_dt", "dtype": "datetime"},
            {"name": "exit_dt", "dtype": "datetime"},
            {"name": "entry_price", "dtype": "float"},
            {"name": "exit_price", "dtype": "float"},
            {"name": "size", "dtype": "float"},
            {"name": "pnl_gross", "dtype": "float"},
            {"name": "pnl_net", "dtype": "float"},
            {"name": "fees", "dtype": "float"},
            {"name": "slippage_entry", "dtype": "float"},
            {"name": "slippage_exit", "dtype": "float"},
            {"name": "stop_price", "dtype": "float"},
            {"name": "target_price", "dtype": "float"},
            {"name": "stop_hit", "dtype": "bool"},
            {"name": "target_hit", "dtype": "bool"},
            {"name": "signal_name", "dtype": "str"},
            {"name": "setup_notes", "dtype": "str"},
            {"name": "entry_reason", "dtype": "str"},
            {"name": "exit_reason", "dtype": "str"},
            {"name": "strategy_version", "dtype": "str"},
            {"name": "run_id", "dtype": "str"},
            {"name": "regime", "dtype": "str"},
            {"name": "filter_flags", "dtype": "str"},
            {"name": "risk_amt", "dtype": "float"},
            {"name": "R_multiple", "dtype": "float"},
            {"name": "percent_equity_risked", "dtype": "float"},
            {"name": "holding_time_mins", "dtype": "float"},
            {"name": "mae", "dtype": "float"},
            {"name": "mfe", "dtype": "float"},
            {"name": "max_position_size_reached", "dtype": "float"},
            {"name": "margin_used_pct", "dtype": "float"},
            {"name": "leverage", "dtype": "float"},
            {"name": "portfolio_heat", "dtype": "float"},
        ],
        "excel": {
            "freeze_header": True,
            "conditional_format": {
                "pnl_net": "green_if_positive_red_if_negative",
                "R_multiple": "3_color_scale",
            },
        },
    },
    "Pair_Summary": {
        "description": "One row per symbol/pair with aggregated performance metrics.",
        "columns": [
            {"name": "pair", "dtype": "str"},
            {"name": "trades", "dtype": "int"},
            {"name": "net_pnl", "dtype": "float"},
            {"name": "win_rate", "dtype": "float"},
            {"name": "profit_factor", "dtype": "float"},
            {"name": "expectancy", "dtype": "float"},
            {"name": "avg_win", "dtype": "float"},
            {"name": "avg_loss", "dtype": "float"},
            {"name": "payoff_ratio", "dtype": "float"},
            {"name": "avg_R", "dtype": "float"},
            {"name": "median_R", "dtype": "float"},
            {"name": "%_R>1", "dtype": "float"},
            {"name": "Sharpe", "dtype": "float"},
            {"name": "Sortino", "dtype": "float"},
            {"name": "Calmar", "dtype": "float"},
            {"name": "max_dd_pct", "dtype": "float"},
            {"name": "max_dd_duration_days", "dtype": "float"},
            {"name": "time_in_drawdown_pct", "dtype": "float"},
            {"name": "best_month", "dtype": "str"},
            {"name": "worst_month", "dtype": "str"},
        ],
        "excel": {
            "note_below_table": "Correlation matrix of per-trade returns goes below this row.",
        },
    },
    "Portfolio_Summary": {
        "description": "Headline stats for the entire portfolio with benchmarks and deltas.",
        "columns": [
            {"name": "metric", "dtype": "str"},
            {"name": "value", "dtype": "float"},
            {"name": "benchmark_buy_and_hold", "dtype": "float"},
            {"name": "benchmark_flat", "dtype": "float"},
            {"name": "strategy_version_delta", "dtype": "float"},
        ],
        "equity_curve": {
            "start_row": 10,
            "columns": [
                {"name": "date", "dtype": "date"},
                {"name": "equity", "dtype": "float"},
                {"name": "drawdown_pct", "dtype": "float"},
                {"name": "benchmark_equity", "dtype": "float"},
            ],
            "chart": {
                "type": "line",
                "x_col": "date",
                "y_col": "equity",
                "insert_cell": "F30",
            },
        },
    },
    "Time_Analysis": {
        "description": "Session, hour-of-day, weekday, and regime performance breakdowns.",
        "sections": {
            "session": {
                "title": "Session breakdown (Tokyo, London, NY)",
                "columns": [
                    {"name": "session", "dtype": "str"},
                    {"name": "trades", "dtype": "int"},
                    {"name": "net_pnl", "dtype": "float"},
                    {"name": "win_rate", "dtype": "float"},
                    {"name": "profit_factor", "dtype": "float"},
                    {"name": "Sharpe", "dtype": "float"},
                    {"name": "avg_R", "dtype": "float"},
                ],
            },
            "hour": {
                "title": "Hour-of-day breakdown (0–23)",
                "columns": [
                    {"name": "hour", "dtype": "int"},
                    {"name": "trades", "dtype": "int"},
                    {"name": "net_pnl", "dtype": "float"},
                    {"name": "win_rate", "dtype": "float"},
                    {"name": "profit_factor", "dtype": "float"},
                    {"name": "Sharpe", "dtype": "float"},
                    {"name": "avg_R", "dtype": "float"},
                ],
            },
            "weekday": {
                "title": "Weekday breakdown (Mon–Sun)",
                "columns": [
                    {"name": "weekday", "dtype": "str"},
                    {"name": "trades", "dtype": "int"},
                    {"name": "net_pnl", "dtype": "float"},
                    {"name": "win_rate", "dtype": "float"},
                    {"name": "profit_factor", "dtype": "float"},
                    {"name": "Sharpe", "dtype": "float"},
                    {"name": "avg_R", "dtype": "float"},
                ],
            },
            "vol_regime": {
                "title": "Volatility / trend regimes (ATR & ADX buckets)",
                "columns": [
                    {"name": "ATR_bucket", "dtype": "str"},
                    {"name": "trend_bucket", "dtype": "str"},
                    {"name": "trades", "dtype": "int"},
                    {"name": "net_pnl", "dtype": "float"},
                    {"name": "win_rate", "dtype": "float"},
                    {"name": "profit_factor", "dtype": "float"},
                    {"name": "avg_R", "dtype": "float"},
                    {"name": "Sharpe", "dtype": "float"},
                ],
            },
        },
    },
    "Pair_Monthly": {
        "description": "Per-pair, per-year, per-month aggregated stats.",
        "columns": [
            {"name": "pair", "dtype": "str"},
            {"name": "year", "dtype": "int"},
            {"name": "month", "dtype": "int"},
            {"name": "trades", "dtype": "int"},
            {"name": "net_pnl", "dtype": "float"},
            {"name": "win_rate", "dtype": "float"},
            {"name": "profit_factor", "dtype": "float"},
            {"name": "expectancy", "dtype": "float"},
            {"name": "Sharpe", "dtype": "float"},
            {"name": "avg_R", "dtype": "float"},
            {"name": "max_dd_pct", "dtype": "float"},
            {"name": "avg_holding_mins", "dtype": "float"},
        ],
    },
    "Risk_Analysis": {
        "description": "R-multiple distribution, drawdowns, slippage, and exit efficiency.",
        "sections": {
            "r_distribution": {
                "title": "R-multiple distribution",
                "columns": [
                    {"name": "R_bin_low", "dtype": "float"},
                    {"name": "R_bin_high", "dtype": "float"},
                    {"name": "count", "dtype": "int"},
                    {"name": "percent_of_trades", "dtype": "float"},
                    {"name": "cum_percent", "dtype": "float"},
                ],
            },
            "drawdowns": {
                "title": "Drawdown episodes",
                "columns": [
                    {"name": "dd_id", "dtype": "int"},
                    {"name": "start_date", "dtype": "date"},
                    {"name": "end_date", "dtype": "date"},
                    {"name": "depth_pct", "dtype": "float"},
                    {"name": "duration_days", "dtype": "float"},
                    {"name": "recovery_days", "dtype": "float"},
                ],
            },
            "slippage": {
                "title": "Slippage analysis (overall or per-pair)",
                "columns": [
                    {"name": "pair", "dtype": "str"},
                    {"name": "avg_slippage_entry", "dtype": "float"},
                    {"name": "avg_slippage_exit", "dtype": "float"},
                    {"name": "%_slippage_gt_5pct_R", "dtype": "float"},
                ],
            },
            "exit_efficiency": {
                "title": "Exit efficiency",
                "columns": [
                    {"name": "pair", "dtype": "str"},
                    {"name": "%_within_10pct_MFE", "dtype": "float"},
                    {"name": "median_MAE", "dtype": "float"},
                    {"name": "median_MFE", "dtype": "float"},
                ],
            },
        },
    },
    "Config": {
        "description": "Run metadata as key/value pairs.",
        "columns": [
            {"name": "key", "dtype": "str"},
            {"name": "value", "dtype": "str"},
        ],
    },
}


class BacktestExcelError(ValueError):
    """Raised when required data is missing or malformed for export."""


def _require_dataframe(value: object, context: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise BacktestExcelError(f"{context} must be provided as a pandas DataFrame.")
    return value


def _validate_columns(df: pd.DataFrame, columns: Sequence[Mapping[str, str]], context: str) -> None:
    required = [col["name"] for col in columns]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise BacktestExcelError(f"Missing columns for {context}: {', '.join(missing)}")


def _write_dataframe(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    start_row: int = 0,
    start_col: int = 0,
) -> None:
    df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=start_col, index=False)


def _write_sectioned_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    sections_schema: Mapping[str, Mapping[str, object]],
    data: Mapping[str, pd.DataFrame],
) -> None:
    row = 0
    for key, schema in sections_schema.items():
        if key not in data:
            raise BacktestExcelError(f"Missing '{key}' section for {sheet_name} sheet.")
        df = _require_dataframe(data[key], f"{sheet_name}.{key}")
        _validate_columns(df, schema["columns"], f"{sheet_name}.{key}")
        _write_dataframe(writer, sheet_name, df, start_row=row + 1)
        worksheet = writer.sheets[sheet_name]
        worksheet.write(row, 0, schema["title"])
        row += len(df) + 3 if len(df) else 2


def write_backtest_excel(
    output_path: str | Path,
    data: Mapping[str, object],
    writer_kwargs: MutableMapping[str, object] | None = None,
) -> None:
    """Write a backtest run to an Excel workbook following ``EXCEL_SCHEMA``.

    Parameters
    ----------
    output_path:
        Destination file path for the generated ``.xlsx`` workbook.
    data:
        Mapping containing pre-calculated data for every sheet defined in
        :data:`EXCEL_SCHEMA`. Expected keys:

        * ``Trades``: DataFrame with per-trade rows.
        * ``Pair_Summary``: DataFrame with aggregated pair metrics.
        * ``Portfolio_Summary``: Either a DataFrame with portfolio metrics or
          a mapping with ``metrics`` and ``equity_curve`` DataFrames. If a
          mapping is used, ``metrics`` holds the main table and
          ``equity_curve`` holds equity curve rows.
        * ``Portfolio_Summary_equity_curve``: Optional DataFrame alternative
          to the nested ``equity_curve`` key.
        * ``Time_Analysis``: Mapping containing the four section DataFrames:
          ``session``, ``hour``, ``weekday``, ``vol_regime``.
        * ``Pair_Monthly``: DataFrame with per-pair per-month aggregates.
        * ``Risk_Analysis``: Mapping with ``r_distribution``, ``drawdowns``,
          ``slippage``, and ``exit_efficiency`` DataFrames.
        * ``Config``: DataFrame with metadata key/value pairs.
    writer_kwargs:
        Additional keyword arguments forwarded to ``pandas.ExcelWriter``.
    """

    writer_kwargs = writer_kwargs or {}
    path = Path(output_path)

    with pd.ExcelWriter(path, engine="xlsxwriter", **writer_kwargs) as writer:
        _write_trades_sheet(writer, data)
        _write_pair_summary_sheet(writer, data)
        _write_portfolio_summary_sheet(writer, data)
        _write_time_analysis_sheet(writer, data)
        _write_pair_monthly_sheet(writer, data)
        _write_risk_analysis_sheet(writer, data)
        _write_config_sheet(writer, data)


def _write_trades_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    df = _require_dataframe(data.get("Trades"), "Trades")
    _validate_columns(df, EXCEL_SCHEMA["Trades"]["columns"], "Trades")
    _write_dataframe(writer, "Trades", df)
    worksheet = writer.sheets["Trades"]
    if EXCEL_SCHEMA["Trades"]["excel"].get("freeze_header"):
        worksheet.freeze_panes(1, 0)

    if len(df):
        workbook = writer.book
        last_row = len(df)
        pnl_col = df.columns.get_loc("pnl_net")
        pos_format = workbook.add_format({"font_color": "#006100"})
        neg_format = workbook.add_format({"font_color": "#9C0006"})
        worksheet.conditional_format(
            1, pnl_col, last_row, pnl_col,
            {"type": "cell", "criteria": ">=", "value": 0, "format": pos_format},
        )
        worksheet.conditional_format(
            1, pnl_col, last_row, pnl_col,
            {"type": "cell", "criteria": "<", "value": 0, "format": neg_format},
        )
        r_col = df.columns.get_loc("R_multiple")
        worksheet.conditional_format(1, r_col, last_row, r_col, {"type": "3_color_scale"})


def _write_pair_summary_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    df = _require_dataframe(data.get("Pair_Summary"), "Pair_Summary")
    _validate_columns(df, EXCEL_SCHEMA["Pair_Summary"]["columns"], "Pair_Summary")
    _write_dataframe(writer, "Pair_Summary", df)
    note = EXCEL_SCHEMA["Pair_Summary"]["excel"]["note_below_table"]
    worksheet = writer.sheets["Pair_Summary"]
    worksheet.write(len(df) + 1, 0, note)


def _write_portfolio_summary_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    portfolio_data = data.get("Portfolio_Summary")
    metrics_df: pd.DataFrame | None = None
    equity_curve_df: pd.DataFrame | None = None

    if isinstance(portfolio_data, pd.DataFrame):
        metrics_df = portfolio_data
    elif isinstance(portfolio_data, Mapping):
        metrics_df = portfolio_data.get("metrics")
        equity_curve_df = portfolio_data.get("equity_curve")

    if equity_curve_df is None and "Portfolio_Summary_equity_curve" in data:
        raw_curve = data.get("Portfolio_Summary_equity_curve")
        if isinstance(raw_curve, pd.DataFrame):
            equity_curve_df = raw_curve

    metrics_df = _require_dataframe(metrics_df, "Portfolio_Summary")
    _validate_columns(metrics_df, EXCEL_SCHEMA["Portfolio_Summary"]["columns"], "Portfolio_Summary")
    _write_dataframe(writer, "Portfolio_Summary", metrics_df)

    if equity_curve_df is not None:
        _write_portfolio_equity_curve(writer, equity_curve_df)


def _write_portfolio_equity_curve(writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
    equity_schema = EXCEL_SCHEMA["Portfolio_Summary"]["equity_curve"]
    _validate_columns(df, equity_schema["columns"], "Portfolio_Summary.equity_curve")
    start_row = equity_schema["start_row"] - 1
    _write_dataframe(writer, "Portfolio_Summary", df, start_row=start_row)

    worksheet = writer.sheets["Portfolio_Summary"]
    workbook = writer.book
    chart_spec = equity_schema["chart"]
    x_col = df.columns.get_loc(chart_spec["x_col"])
    y_col = df.columns.get_loc(chart_spec["y_col"])
    last_row = start_row + len(df)

    chart = workbook.add_chart({"type": chart_spec["type"]})
    chart.add_series({
        "name": "Equity",
        "categories": ["Portfolio_Summary", start_row + 1, x_col, last_row, x_col],
        "values": ["Portfolio_Summary", start_row + 1, y_col, last_row, y_col],
    })
    worksheet.insert_chart(chart_spec["insert_cell"], chart)


def _write_time_analysis_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    sections_data = data.get("Time_Analysis")
    if not isinstance(sections_data, Mapping):
        raise BacktestExcelError("Time_Analysis sections must be provided as a mapping of DataFrames.")
    _write_sectioned_sheet(writer, "Time_Analysis", EXCEL_SCHEMA["Time_Analysis"]["sections"], sections_data)


def _write_pair_monthly_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    df = _require_dataframe(data.get("Pair_Monthly"), "Pair_Monthly")
    _validate_columns(df, EXCEL_SCHEMA["Pair_Monthly"]["columns"], "Pair_Monthly")
    _write_dataframe(writer, "Pair_Monthly", df)


def _write_risk_analysis_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    sections_data = data.get("Risk_Analysis")
    if not isinstance(sections_data, Mapping):
        raise BacktestExcelError("Risk_Analysis sections must be provided as a mapping of DataFrames.")
    _write_sectioned_sheet(writer, "Risk_Analysis", EXCEL_SCHEMA["Risk_Analysis"]["sections"], sections_data)


def _write_config_sheet(writer: pd.ExcelWriter, data: Mapping[str, object]) -> None:
    df = _require_dataframe(data.get("Config"), "Config")
    _validate_columns(df, EXCEL_SCHEMA["Config"]["columns"], "Config")
    _write_dataframe(writer, "Config", df)