"""Align Tardis snapshot and trade CSVs using local receipt timestamps.

Trade events at exactly a snapshot's timestamp are conservatively excluded;
cross-file ordering of tied timestamps is not assumed to be known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ts.data.lob_processor import LOBWindowProcessor


def snapshot_trade_features(snapshots: pd.DataFrame, trades: pd.DataFrame):
    """Build features with observable trades aggregated between snapshot updates.

    The direction channel is signed-volume / total-volume, rather than a single
    trade's binary sign. This preserves signed volume when multiple trades fall
    between snapshots, and is a documented sampling difference from trade ticks.
    """
    books = snapshots.sort_values("local_timestamp", kind="stable").reset_index(drop=True)
    fills = trades.sort_values("local_timestamp", kind="stable").reset_index(drop=True)
    if books.empty or fills.empty:
        raise ValueError("Both real snapshots and trades are required")
    st = pd.to_numeric(books.local_timestamp, errors="raise").to_numpy(dtype=np.int64)
    tt = pd.to_numeric(fills.local_timestamp, errors="raise").to_numpy(dtype=np.int64)
    amounts = pd.to_numeric(fills.amount, errors="raise").to_numpy(dtype=float)
    if (amounts < 0).any() or not np.isfinite(amounts).all():
        raise ValueError("Trade amounts must be nonnegative and finite")
    side = fills.side.astype(str).str.lower()
    if not side.isin(["buy", "sell"]).all():
        raise ValueError("Trade sides must be buy or sell")
    signed = amounts * np.where(side.eq("buy"), 1., -1.)
    positions = np.searchsorted(tt, st, side="left")
    total_before = np.concatenate([[0.], np.cumsum(amounts)])[positions]
    signed_before = np.concatenate([[0.], np.cumsum(signed)])[positions]
    volume = np.diff(total_before, prepend=0.)
    signed_volume = np.diff(signed_before, prepend=0.)
    direction = np.divide(signed_volume, volume, out=np.zeros_like(volume), where=volume > 0)
    # Roundoff in cumulative sums can exceed one by tiny amounts.
    direction = np.clip(direction, -1., 1.)
    def levels(side_name, field):
        return books[[f"{side_name}[{i}].{field}" for i in range(10)]].to_numpy(dtype=float)
    bid, ask = levels("bids", "price"), levels("asks", "price")
    bv, av = levels("bids", "amount"), levels("asks", "amount")
    if not all(np.isfinite(x).all() for x in [bid, ask, bv, av]):
        raise ValueError("Missing or nonfinite order book levels")
    if (bid <= 0).any() or (ask <= 0).any() or (bv < 0).any() or (av < 0).any():
        raise ValueError("Invalid order book prices or amounts")
    if (bid[:, 0] >= ask[:, 0]).any():
        raise ValueError("Crossed/locked best bid and ask require source review")
    features = LOBWindowProcessor().compute_40_lob_features(bid, bv, ask, av, volume, direction)
    return pd.to_datetime(st, unit="us", utc=True), features


def read_snapshot_trade_features(book_path, trade_path):
    book_columns = ["local_timestamp"] + [f"{side}[{i}].{field}"
                    for i in range(10) for side in ["bids", "asks"] for field in ["price", "amount"]]
    snapshots = pd.read_csv(book_path, usecols=book_columns)
    trades = pd.read_csv(trade_path, usecols=["local_timestamp", "amount", "side"])
    return snapshot_trade_features(snapshots, trades)
