"""
API-Sports F1 client — for dashboard UI assets (images, logos, metadata).

Not used for prediction model — data overlaps with Jolpica (and only 2012+).
Primary value: driver headshot photos, team logos, circuit images, widgets.

Free plan: 100 requests/day. Images/logos don't count toward quota.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://v1.formula-1.api-sports.io"
CACHE_DIR = Path(__file__).parent.parent / "cache" / "apisports"

# Endpoints that should never be cached (live/dynamic data)
_NO_CACHE_ENDPOINTS = {"status"}


class APISportsF1Client:
    """Client for API-Sports Formula 1 API — focused on UI assets."""

    def __init__(self, api_key: Optional[str] = None, cache: bool = True):
        self.api_key = api_key or os.getenv("API_SPORTS_KEY", "")
        if not self.api_key:
            raise ValueError(
                "API_SPORTS_KEY not set. Add it to .env or pass api_key param."
            )
        self.session = requests.Session()
        self.session.headers["x-apisports-key"] = self.api_key
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a cached API request."""
        params = params or {}
        skip_cache = endpoint.strip("/") in _NO_CACHE_ENDPOINTS

        if self.cache and not skip_cache:
            raw = endpoint.strip("/") + "?" + "&".join(
                f"{k}={v}" for k, v in sorted(params.items())
            )
            cache_key = hashlib.sha256(raw.encode()).hexdigest()[:16]
            cache_file = CACHE_DIR / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        else:
            cache_file = None

        resp = self.session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if cache_file is not None:
            with open(cache_file, "w") as f:
                json.dump(data, f)

        time.sleep(0.3)
        return data

    # ── UI Asset Endpoints ──────────────────────────────────────────

    def get_circuits(self) -> list[dict]:
        """All circuits with images, lap records, capacity, track length."""
        data = self._get("circuits")
        return data.get("response", [])

    def get_teams(self, name: Optional[str] = None) -> list[dict]:
        """Teams with logos, base, championships, technical details."""
        params = {}
        if name:
            params["search"] = name
        data = self._get("teams", params)
        return data.get("response", [])

    def get_drivers(self, search: Optional[str] = None) -> list[dict]:
        """
        Drivers with photos, career stats.
        Requires at least one param (id, name, or search).
        """
        if not search:
            raise ValueError("API-Sports /drivers requires at least one parameter (search, name, or id)")
        data = self._get("drivers", {"search": search})
        return data.get("response", [])

    def get_driver_standings(self, season: int) -> list[dict]:
        """Driver championship standings with photos and team logos."""
        data = self._get("rankings/drivers", {"season": str(season)})
        return data.get("response", [])

    def get_team_standings(self, season: int) -> list[dict]:
        """Constructor championship standings with team logos."""
        data = self._get("rankings/teams", {"season": str(season)})
        return data.get("response", [])

    def get_races(self, season: int, race_type: str = "Race") -> list[dict]:
        """Race schedule with circuit images and fastest lap info."""
        data = self._get("races", {"season": str(season), "type": race_type})
        return data.get("response", [])

    def get_race_results(self, race_id: int) -> list[dict]:
        """Race results with driver photos and team logos."""
        data = self._get("rankings/races", {"race": str(race_id)})
        return data.get("response", [])

    def get_fastest_laps(self, race_id: int) -> list[dict]:
        """Fastest lap rankings for a race."""
        data = self._get("rankings/fastestlaps", {"race": str(race_id)})
        return data.get("response", [])

    def get_starting_grid(self, race_id: int) -> list[dict]:
        """Starting grid for a race."""
        data = self._get("rankings/startinggrid", {"race": str(race_id)})
        return data.get("response", [])

    def get_pit_stops(self, race_id: int) -> list[dict]:
        """Pit stop data for a race."""
        data = self._get("pitstops", {"race": str(race_id)})
        return data.get("response", [])

    def get_status(self) -> dict:
        """Check account status and remaining daily requests."""
        data = self._get("status")
        return data.get("response", {})

    # ── Asset URL Helpers ───────────────────────────────────────────

    @staticmethod
    def driver_image_url(driver_id: int) -> str:
        """URL for driver headshot photo (doesn't count toward quota)."""
        return f"https://media.api-sports.io/formula-1/drivers/{driver_id}.png"

    @staticmethod
    def team_logo_url(team_id: int) -> str:
        """URL for team logo (doesn't count toward quota)."""
        return f"https://media.api-sports.io/formula-1/teams/{team_id}.png"

    @staticmethod
    def circuit_image_url(circuit_id: int) -> str:
        """URL for circuit image (doesn't count toward quota)."""
        return f"https://media.api-sports.io/formula-1/circuits/{circuit_id}.png"

    # ── Bulk Cache ──────────────────────────────────────────────────

    def cache_season_assets(self, season: int) -> dict:
        """
        Cache all UI assets for a season in one batch.
        Uses ~3-4 API calls (standings + races).
        Returns asset URLs for the dashboard.
        """
        assets = {"drivers": {}, "teams": {}, "circuits": {}}

        # Driver standings → driver photos + team logos
        standings = self.get_driver_standings(season)
        for s in standings:
            driver = s.get("driver", {})
            team = s.get("team", {})
            did = driver.get("id")
            tid = team.get("id")
            if did:
                assets["drivers"][driver.get("abbr", driver.get("name", ""))] = {
                    "id": did,
                    "name": driver.get("name"),
                    "image": driver.get("image"),
                    "abbr": driver.get("abbr"),
                }
            if tid:
                assets["teams"][team.get("name", "")] = {
                    "id": tid,
                    "name": team.get("name"),
                    "logo": team.get("logo"),
                }

        # Races → circuit images
        races = self.get_races(season)
        for r in races:
            circuit = r.get("circuit", {})
            cid = circuit.get("id")
            if cid:
                assets["circuits"][circuit.get("name", "")] = {
                    "id": cid,
                    "name": circuit.get("name"),
                    "image": circuit.get("image"),
                }

        logger.info(
            f"Cached {season} assets: "
            f"{len(assets['drivers'])} drivers, "
            f"{len(assets['teams'])} teams, "
            f"{len(assets['circuits'])} circuits"
        )
        return assets

    # ── Widget HTML Generator ───────────────────────────────────────

    def widget_html(
        self,
        widget_type: str = "races",
        season: Optional[int] = None,
        theme: str = "dark",
        **kwargs,
    ) -> str:
        """
        Generate HTML snippet for API-Sports F1 widget.

        Security: API key is embedded in the HTML (visible in browser DOM).
        This is by design per API-Sports docs. Restrict allowed domains in
        the API-Sports dashboard to prevent unauthorized key usage.

        widget_type: "races", "race", "driver"
        theme: "white", "grey", "dark", "blue", or custom CSS theme name
        """
        attrs = [
            f'data-type="{widget_type}"',
            f'data-theme="{theme}"',
        ]
        if season:
            attrs.append(f'data-season="{season}"')
        for key, val in kwargs.items():
            attrs.append(f'data-{key.replace("_", "-")}="{val}"')

        widget_tag = f'<api-sports-widget {" ".join(attrs)}></api-sports-widget>'
        config_tag = (
            f'<api-sports-widget data-type="config" '
            f'data-key="{self.api_key}" '
            f'data-sport="formula-1" '
            f'data-theme="{theme}"'
            f'></api-sports-widget>'
        )
        script_tag = '<script src="https://widgets.api-sports.io/2.0.3/widget.js"></script>'

        return f"{script_tag}\n{widget_tag}\n{config_tag}"


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    client = APISportsF1Client()

    status = client.get_status()
    print(f"Account: {status.get('account', {}).get('firstname', 'N/A')}")
    print(f"Plan: {status.get('subscription', {}).get('plan', 'N/A')}")
    print(f"Requests: {status.get('requests', {})}")

    assets = client.cache_season_assets(2024)
    for category, items in assets.items():
        print(f"\n{category}: {len(items)} items")
        for name, info in list(items.items())[:3]:
            print(f"  {name}: {info}")
