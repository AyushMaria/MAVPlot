"""
llm/log_extractor.py

Headless database-query layer for MAVLink flight logs.

Responsibilities
----------------
1. Parse a MAVLink .tlog / .bin / .log file into a dict of
   per-message-type pandas DataFrames, each with a monotonic
   ``time_s`` column (seconds from first message).
2. Provide a lightweight schema-only pass that returns message-type
   metadata without materialising full DataFrames (used for embeddings).
3. Export a subset of message types to a single Parquet file so the
   LLM only sees a clean, time-aligned, typed tabular structure —
   never raw binary data.

Design notes
------------
* Each DataFrame has a ``msg_type`` string column and a float64
  ``time_s`` column derived from the MAVLink timestamp (us_since_epoch
  when available, else monotonic sequence * 0.001 s).
* All numeric columns are cast to float64 for plotting compatibility.
* Non-numeric, non-string fields are dropped (arrays, structs).
* The Parquet file uses per-column statistics so the LLM prompt can
  include min/max/dtype metadata without reading the file itself.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from pymavlink import mavutil

logger = logging.getLogger(__name__)

# Columns injected by the extractor — not sourced from the MAVLink message.
_RESERVED_COLS = {"time_s", "msg_type"}


def _to_float(value) -> Optional[float]:
    """Convert a scalar to float64; return None if not convertible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_us(msg) -> Optional[int]:
    """
    Extract a microsecond-epoch timestamp from a MAVLink message.

    Tries common timestamp field names in priority order.
    Returns None when no timestamp field is present.
    """
    for attr in ("time_usec", "time_unix_usec", "time_us", "usec"):
        val = getattr(msg, attr, None)
        if val and val > 0:
            return int(val)
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to coerce a Series to numeric (float64).

    Uses errors='coerce' (pandas >= 2.0 dropped 'ignore').
    If the result is all-NaN the original string Series is returned
    unchanged so we don't silently destroy text columns.
    """
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().all():
        return series  # preserve string / mixed columns as-is
    return converted


class LogExtractor:
    """
    Parse a MAVLink log file into structured DataFrames.

    Parameters
    ----------
    log_path:
        Absolute or relative path to a .tlog / .bin / .log file.
    """

    def __init__(self, log_path: str) -> None:
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        self.log_path = log_path
        # Populated after extract_all()
        self._frames: Dict[str, pd.DataFrame] = {}
        self._schema: Dict[str, dict] = {}   # msg_type -> {field: dtype_str}
        self._extracted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schema_only(self) -> Dict[str, dict]:
        """
        Fast single-pass scan that collects message-type metadata
        (field names, Python types, message count) without storing rows.

        Returns
        -------
        dict mapping msg_type -> {"count": int, "fields": {name: type_str}}
        """
        schema: Dict[str, dict] = {}
        mav = mavutil.mavlink_connection(self.log_path)
        while True:
            try:
                msg = mav.recv_match(blocking=False)
                if msg is None:
                    break
                mt = msg.get_type()
                if mt == "BAD_DATA":
                    continue
                if mt not in schema:
                    schema[mt] = {
                        "count": 1,
                        "fields": {
                            f: type(getattr(msg, f)).__name__
                            for f in msg.get_fieldnames()
                        },
                    }
                else:
                    schema[mt]["count"] += 1
            except KeyboardInterrupt:
                logger.info("schema_only() interrupted.")
                break
            except Exception as exc:
                logger.debug("schema_only skip: %s", exc)
                break
        self._schema = schema
        return schema

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """
        Full two-pass extraction.

        Pass 1 — collect raw rows per message type.
        Pass 2 — build DataFrames, normalise time_s, cast numerics.

        Returns
        -------
        dict mapping msg_type -> pd.DataFrame with columns:
            time_s (float64), msg_type (str), <field_0>, <field_1>, ...
        """
        raw: Dict[str, list] = {}  # msg_type -> list of row-dicts
        t0_us: Optional[int] = None
        seq = 0  # fallback counter when no timestamp

        mav = mavutil.mavlink_connection(self.log_path)
        while True:
            try:
                msg = mav.recv_match(blocking=False)
                if msg is None:
                    break
                mt = msg.get_type()
                if mt == "BAD_DATA":
                    continue

                # --- timestamp resolution ---
                ts_us = _timestamp_us(msg)
                if ts_us is not None:
                    if t0_us is None:
                        t0_us = ts_us
                    time_s = (ts_us - t0_us) / 1_000_000.0
                else:
                    time_s = seq * 0.001
                seq += 1

                row: dict = {"time_s": time_s, "msg_type": mt}
                for field in msg.get_fieldnames():
                    val = getattr(msg, field, None)
                    # Keep scalars only — drop lists / nested objects
                    if isinstance(val, (int, float, str, bool)):
                        row[field] = val

                raw.setdefault(mt, []).append(row)

            except KeyboardInterrupt:
                logger.info("extract_all() interrupted.")
                break
            except Exception as exc:
                logger.debug("extract_all skip msg: %s", exc)
                break

        # Build DataFrames
        frames: Dict[str, pd.DataFrame] = {}
        for mt, rows in raw.items():
            df = pd.DataFrame(rows)
            # Coerce numeric columns to float64, preserve string cols
            for col in df.columns:
                if col not in _RESERVED_COLS:
                    df[col] = _coerce_numeric(df[col])
            df.sort_values("time_s", inplace=True)
            df.reset_index(drop=True, inplace=True)
            frames[mt] = df

        self._frames = frames
        self._extracted = True
        logger.info("Extracted %d message types from %s", len(frames), self.log_path)
        return frames

    def export_parquet(
        self,
        msg_types: List[str],
        output_path: str,
    ) -> Dict[str, dict]:
        """
        Export the requested message types to a single Parquet file.

        Each message type becomes a separate row-group distinguished
        by the ``msg_type`` column.  Only types actually present in the
        log are included (unknown types are silently skipped).

        Parameters
        ----------
        msg_types:
            List of MAVLink message type names to export, e.g.
            ["GLOBAL_POSITION_INT", "SYS_STATUS"].
        output_path:
            Destination .parquet file path.

        Returns
        -------
        Schema summary dict mapping msg_type ->
            {"rows": int, "columns": [col_name: dtype_str, ...]}
        Suitable for injection into the LLM prompt.
        """
        if not self._extracted:
            raise RuntimeError("Call extract_all() before export_parquet().")

        selected = [
            self._frames[mt]
            for mt in msg_types
            if mt in self._frames
        ]
        skipped = [mt for mt in msg_types if mt not in self._frames]
        if skipped:
            logger.warning("Requested types not found in log: %s", skipped)

        if not selected:
            raise ValueError(
                f"None of the requested message types were found in the log. "
                f"Requested: {msg_types}"
            )

        combined = pd.concat(selected, ignore_index=True)
        combined.sort_values("time_s", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Exported %d rows to %s", len(combined), output_path)

        # Build schema summary for LLM prompt
        summary: Dict[str, dict] = {}
        for mt in msg_types:
            if mt not in self._frames:
                continue
            df = self._frames[mt]
            cols = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                if pd.api.types.is_numeric_dtype(df[col]):
                    cols[col] = {
                        "dtype": dtype,
                        "min": round(float(df[col].min()), 6),
                        "max": round(float(df[col].max()), 6),
                    }
                else:
                    cols[col] = {"dtype": dtype}
            summary[mt] = {"rows": len(df), "columns": cols}

        return summary

    @property
    def frames(self) -> Dict[str, pd.DataFrame]:
        """The extracted frames dict. Empty until extract_all() is called."""
        return self._frames
