"""
Report generator for StaticBacktester results.

Outputs:
  1. Rich terminal tables (Summary stats + Trade log) printed to console.
  2. Paginated PNG snapshots -- up to batch_size trades per image (default 20).
     Page 1 includes the summary table. Files are auto-numbered:
     {base}_table_01.png, {base}_table_02.png ...
  3. CSV trade log saved alongside the PNGs.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from .config import BacktestConfig
    from .engine import BacktestResult

logger = logging.getLogger(__name__)

_GREEN  = "#00e676"
_RED    = "#ff1744"
_GREY   = "#90a4ae"
_YELLOW = "#FFD600"
_CYAN   = "#26C6DA"


def _pnl_color(val: float) -> str:
    return _GREEN if val > 0 else (_RED if val < 0 else _GREY)


def generate_report(
    result: "BacktestResult",
    symbol: str,
    timeframe: str,
    config: "BacktestConfig",
    output_dir: Path,
    batch_size: int = 20,
) -> Tuple[Path, List[Path]]:
    """Generate CSV trade log + paginated PNG table images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{symbol}_{timeframe}_{ts}"
    csv_path = output_dir / f"{base}_trades.csv"

    from rich.console import Console

    summary_table = _build_summary_table(result, symbol, timeframe, config)

    trades = result.trades
    batches: list[list[Any]] = []
    if trades:
        for start in range(0, len(trades), batch_size):
            end = min(start + batch_size, len(trades))
            chunk = [(start + j + 1, trades[start + j]) for j in range(end - start)]
            batches.append(chunk)
    if not batches:
        batches = [[]]

    total_pages = len(batches)

    # --- Print full table to terminal (unchanged UX) ---
    full_trade_table = _build_trade_table_rows(
        [(i + 1, t) for i, t in enumerate(trades)],
        page_label=None,
    )
    console_live = Console()
    console_live.print()
    console_live.print(summary_table)
    console_live.print()
    console_live.print(full_trade_table)
    console_live.print()

    # --- Save CSV ---
    _save_csv(result, csv_path)
    logger.info("Trade log CSV -> %s", csv_path)

    # --- Save paginated PNGs ---
    table_png_paths: List[Path] = []
    for page_num, batch in enumerate(batches, start=1):
        png_path = output_dir / f"{base}_table_{page_num:02d}.png"
        page_label = f"Page {page_num}/{total_pages}"
        trade_table = _build_trade_table_rows(batch, page_label=page_label)

        console_rec = Console(record=True, width=140)
        if page_num == 1:
            console_rec.print(summary_table)
            console_rec.print()
        console_rec.print(trade_table)

        svg_title = f"Backtest Report - {symbol} / {timeframe}  [{page_label}]"
        svg_str = console_rec.export_svg(title=svg_title)
        _svg_to_png(svg_str, png_path)
        table_png_paths.append(png_path)
        logger.info("Table PNG (%s) -> %s", page_label, png_path)

    return csv_path, table_png_paths


def _build_summary_table(
    result: "BacktestResult",
    symbol: str,
    timeframe: str,
    config: "BacktestConfig",
) -> Any:
    from rich import box
    from rich.table import Table

    m = result.metrics
    cfg = config

    t = Table(
        title=f"[bold cyan]Backtest Summary[/]  [dim]{symbol} / {timeframe}[/]  [dim]{cfg.summary()}[/]",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        show_lines=False,
        min_width=110,
    )

    cols = [
        ("Trades",        str(int(m["num_trades"]))),
        ("Win Rate",      f"{m['win_rate']*100:.1f}%"),
        ("Total Return",  _colored(f"{m['total_return_pct']:+.2f}%",  _pnl_color(m["total_return_pct"]))),
        ("Profit Factor", _colored(f"{m['profit_factor']:.2f}",       _GREEN if m["profit_factor"] > 1 else _RED)),
        ("Max DD",        _colored(f"{m['max_drawdown_pct']:.2f}%",   _RED if m["max_drawdown_pct"] > 5 else _GREY)),
        ("Sharpe",        f"{m['sharpe_ratio']:.2f}"),
        ("Avg R/R",       f"{m['avg_rr']:.2f}"),
        ("Avg Bars",      f"{m['avg_bars_held']:.1f}"),
        ("Long",          _colored(str(int(m["num_long"])),  _GREEN)),
        ("Short",         _colored(str(int(m["num_short"])), _RED)),
        ("Best Trade",    _colored(f"{m['best_trade_pct']:+.2f}%",    _GREEN)),
        ("Worst Trade",   _colored(f"{m['worst_trade_pct']:+.2f}%",   _RED)),
    ]

    for header, _ in cols:
        t.add_column(header, justify="right", no_wrap=True)

    t.add_row(*[v for _, v in cols])
    return t


def _build_trade_table_rows(
    numbered_trades: list[Any],
    page_label: str | None = None,
) -> Any:
    from rich import box
    from rich.table import Table

    title = "[bold cyan]Trade Log[/]"
    if page_label:
        title += f"  [dim]{page_label}[/]"

    t = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        header_style="bold dim",
        show_lines=True,
        min_width=110,
    )
    t.add_column("#",          justify="right",  style="dim",        width=4)
    t.add_column("Dir",        justify="center", no_wrap=True,       width=6)
    t.add_column("Entry Time", justify="left",   style="dim",        width=20)
    t.add_column("Exit Time",  justify="left",   style="dim",        width=20)
    t.add_column("Entry $",    justify="right",  no_wrap=True,       width=12)
    t.add_column("Exit $",     justify="right",  no_wrap=True,       width=12)
    t.add_column("TP $",       justify="right",  style="dark_green", width=12)
    t.add_column("SL $",       justify="right",  style="red",        width=12)
    t.add_column("Reason",     justify="center", no_wrap=True,       width=7)
    t.add_column("PnL%",       justify="right",  no_wrap=True,       width=9)
    t.add_column("Bars",       justify="right",  style="dim",        width=5)

    if not numbered_trades:
        t.add_row(*(["—"] * 11))
        return t

    for idx, trade in numbered_trades:
        dir_str = (
            _colored("LONG▲", _GREEN) if trade.direction == "LONG"
            else _colored("SHORT▼", _RED)
        )
        reason_color = {
            "TP":     _GREEN,
            "SL":     _RED,
            "TRAIL":  _YELLOW,
            "TIME":   _GREY,
            "SIGNAL": _CYAN,
        }.get(trade.exit_reason, _GREY)

        t.add_row(
            str(idx),
            dir_str,
            str(trade.time_entry)[:19],
            str(trade.time_exit)[:19],
            f"{trade.entry_price:,.4f}",
            f"{trade.exit_price:,.4f}",
            f"{trade.tp_price:,.4f}",
            f"{trade.sl_price:,.4f}",
            _colored(trade.exit_reason, reason_color),
            _colored(f"{trade.pnl_pct:+.2f}%", _pnl_color(trade.pnl_pct)),
            str(trade.bars_held),
        )

    return t


def _save_csv(result: "BacktestResult", path: Path) -> None:
    fieldnames = [
        "num", "direction", "time_entry", "time_exit",
        "entry_price", "exit_price", "tp_price", "sl_price",
        "exit_reason", "pnl_pct", "bars_held",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, tr in enumerate(result.trades, 1):
            writer.writerow({
                "num":         i,
                "direction":   tr.direction,
                "time_entry":  str(tr.time_entry)[:19],
                "time_exit":   str(tr.time_exit)[:19],
                "entry_price": f"{tr.entry_price:.6f}",
                "exit_price":  f"{tr.exit_price:.6f}",
                "tp_price":    f"{tr.tp_price:.6f}",
                "sl_price":    f"{tr.sl_price:.6f}",
                "exit_reason": tr.exit_reason,
                "pnl_pct":     f"{tr.pnl_pct:.4f}",
                "bars_held":   tr.bars_held,
            })


def _svg_to_png(svg_str: str, output_path: Path) -> None:
    try:
        import cairosvg  # type: ignore[import-untyped]
        cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            write_to=str(output_path),
            scale=2,
        )
    except Exception as exc:
        svg_path = output_path.with_suffix(".svg")
        svg_path.write_text(svg_str, encoding="utf-8")
        logger.warning(
            "cairosvg conversion failed (%s); saved SVG instead -> %s", exc, svg_path
        )


def _colored(text: str, color: str) -> str:
    return f"[{color}]{text}[/{color}]"
