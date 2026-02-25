"""EIA data client with local caching for WTI Cushing daily spot prices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

EIA_API_V2_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
WTI_PRODUCT_CODE = "EPCWTI"


class DataFetchError(RuntimeError):
    """Raised when upstream data cannot be fetched and no cache is available."""


@dataclass
class EIAClientConfig:
    """Runtime configuration for the EIA WTI client."""

    cache_dir: Path = Path("data/raw")
    cache_filename: str = "wti_cushing_daily.csv"
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    page_size: int = 5000


class EIAWTIClient:
    """Fetches and caches EIA WTI Cushing daily spot prices."""

    def __init__(self, config: Optional[EIAClientConfig] = None) -> None:
        self.config = config or EIAClientConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.config.cache_dir / self.config.cache_filename

    def fetch_daily_wti(
        self,
        force_refresh: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch WTI daily price series with cache fallback.

        Args:
            force_refresh: If True, always fetches from API and overwrites cache.
            start_date: Optional inclusive start date in YYYY-MM-DD.
            end_date: Optional inclusive end date in YYYY-MM-DD.

        Returns:
            DataFrame with columns: date (datetime64), price (float).
        """
        if self.cache_path.exists() and not force_refresh:
            return self._load_cache(start_date=start_date, end_date=end_date)

        try:
            df = self._fetch_from_api(start_date=start_date, end_date=end_date)
            self._write_cache(df)
            return df
        except Exception as exc:  # pragma: no cover - exercised via fallback test path
            if self.cache_path.exists():
                return self._load_cache(start_date=start_date, end_date=end_date)
            raise DataFetchError(
                "Unable to fetch EIA WTI data and no cache is available."
            ) from exc

    def _fetch_from_api(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        api_key = self.config.api_key or os.getenv("EIA_API_KEY", "DEMO_KEY")
        rows: list[dict[str, Any]] = []
        offset = 0
        total_rows: Optional[int] = None

        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            while True:
                params: dict[str, Any] = {
                    "api_key": api_key,
                    "frequency": "daily",
                    "data[0]": "value",
                    "facets[product][]": WTI_PRODUCT_CODE,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                    "offset": offset,
                    "length": self.config.page_size,
                }
                if start_date:
                    params["start"] = start_date
                if end_date:
                    params["end"] = end_date

                response = client.get(EIA_API_V2_URL, params=params)
                response.raise_for_status()
                payload = response.json()

                response_obj = payload.get("response", {})
                batch = response_obj.get("data", [])
                if total_rows is None:
                    try:
                        total_rows = int(response_obj.get("total", 0))
                    except (TypeError, ValueError):
                        total_rows = 0

                if not batch:
                    break

                rows.extend(batch)
                offset += len(batch)

                if total_rows and offset >= total_rows:
                    break

        if not rows:
            raise DataFetchError("EIA API returned no rows for the requested range.")

        parsed_rows = []
        for row in rows:
            value = row.get("value")
            period = row.get("period")
            if value in (None, "") or period in (None, ""):
                continue
            parsed_rows.append({"date": period, "price": float(value)})

        df = pd.DataFrame(parsed_rows)
        if df.empty:
            raise DataFetchError("EIA API returned empty rows after parsing.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "price"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = df.reset_index(drop=True)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def _write_cache(self, df: pd.DataFrame) -> None:
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(self.cache_path, index=False)

    def _load_cache(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        if not self.cache_path.exists():
            raise DataFetchError(f"Cache file not found: {self.cache_path}")

        df = pd.read_csv(self.cache_path)
        if "date" not in df.columns or "price" not in df.columns:
            raise DataFetchError("Cache file is missing required columns: date, price")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "price"])

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        return df.sort_values("date").reset_index(drop=True)
