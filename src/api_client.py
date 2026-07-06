"""Wrapped NBA API client with disk caching, throttling, and retries.

Every call goes through ``NBAApiClient.fetch``, which:
  1. Checks for a cached JSON file at ``data/raw/<endpoint>/<hash>.json``.
  2. If absent, throttles (sleeps until REQUEST_DELAY has elapsed since the
     last request), calls the fetcher function, and writes the result to disk.
  3. Wraps the call in tenacity exponential-backoff retries.

This means re-running any pipeline step is safe: cached responses are served
from disk instantly, and only missing ones hit the network.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src import config

logger = logging.getLogger(__name__)


class NBAApiClient:
    def __init__(
        self,
        raw_dir: Path = config.RAW_DIR,
        request_delay: float = config.REQUEST_DELAY,
    ):
        self.raw_dir = Path(raw_dir)
        self.request_delay = request_delay
        self._last_request_time = 0.0

    # ---- caching ----

    @staticmethod
    def _cache_key(params: dict) -> str:
        """Stable filename-safe hash of the request parameters."""
        payload = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def _cache_path(self, endpoint_name: str, params: dict) -> Path:
        directory = self.raw_dir / endpoint_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self._cache_key(params)}.json"

    # ---- throttling ----

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = self.request_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    # ---- public API ----

    def fetch(
        self,
        endpoint_name: str,
        fetcher_fn: Callable[[], Any],
        params: dict,
        force: bool = False,
    ) -> dict:
        """Fetch a response, using disk cache when possible.

        Args:
            endpoint_name: e.g. ``"LeagueGameLog"``. Determines the cache subfolder.
            fetcher_fn: zero-arg callable returning the raw JSON dict.
            params: request parameters; used to derive the cache key.
            force: if True, bypass the cache and re-fetch.
        """
        path = self._cache_path(endpoint_name, params)

        if path.exists() and not force:
            with open(path, "r") as f:
                return json.load(f)

        self._throttle()
        data = self._fetch_with_retry(endpoint_name, fetcher_fn)

        # Atomic write: tmp file then rename, so a crash mid-write doesn't leave junk.
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(path)
        return data

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.RETRY_BACKOFF_MULTIPLIER,
            max=config.RETRY_BACKOFF_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_with_retry(self, endpoint_name: str, fetcher_fn: Callable[[], Any]) -> dict:
        try:
            return fetcher_fn()
        except Exception as e:
            logger.warning("API call to %s failed: %s", endpoint_name, e)
            raise
