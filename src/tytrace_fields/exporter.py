from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset, num2date

from .interpolation import bilinear_resample, build_axis


@dataclass(frozen=True)
class ExportResult:
    typhoon: str
    output_directory: Path
    frame_count: int


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _netcdf_datetime(value: Any) -> datetime:
    return datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        tzinfo=UTC,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(temporary, path)


def _grid_geojson(
    timestamp: str,
    bounds: list[float],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    unit: str,
    values: dict[str, np.ndarray],
) -> dict[str, Any]:
    grid: dict[str, Any] = {
        "timestamp": timestamp,
        "bounds": bounds,
        "width": int(longitudes.size),
        "height": int(latitudes.size),
        "longitudeStep": float(longitudes[1] - longitudes[0]) if longitudes.size > 1 else 0,
        "latitudeStep": float(latitudes[1] - latitudes[0]) if latitudes.size > 1 else 0,
        "origin": "north-west",
        "order": "row-major",
        "unit": unit,
    }
    grid.update({name: value.ravel().tolist() for name, value in values.items()})
    return {"type": "FeatureCollection", "features": [], "grid": grid}


def export_typhoon(
    dataset_root: Path,
    output_root: Path,
    typhoon: str,
    resolution: float = 0.25,
    overwrite: bool = False,
) -> ExportResult:
    source_directory = dataset_root / typhoon
    netcdf_path = source_directory / f"{typhoon}.nc"
    info_path = source_directory / "info.json"
    if not netcdf_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"dataset is incomplete: {source_directory}")

    output_directory = output_root / typhoon
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_directory}; pass --overwrite")

    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    start_time = _utc_datetime(info["start_time"])
    end_time = _utc_datetime(info["end_time"])

    frame_entries: list[dict[str, str]] = []
    pressure_min = np.inf
    pressure_max = -np.inf
    wind_min = np.inf
    wind_max = -np.inf

    with Dataset(netcdf_path) as dataset:
        required = {"valid_time", "latitude", "longitude", "msl", "u10", "v10"}
        missing = required.difference(dataset.variables)
        if missing:
            raise ValueError(f"NetCDF is missing variables: {', '.join(sorted(missing))}")

        source_latitudes = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        source_longitudes = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
        north = float(np.max(source_latitudes))
        south = float(np.min(source_latitudes))
        west = float(np.min(source_longitudes))
        east = float(np.max(source_longitudes))
        target_latitudes = build_axis(south, north, resolution)[::-1]
        target_longitudes = build_axis(west, east, resolution)
        bounds = [west, south, east, north]

        time_variable = dataset.variables["valid_time"]
        calendar = getattr(time_variable, "calendar", "standard")
        decoded_times = num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
        )

        for index, raw_time in enumerate(decoded_times):
            frame_time = _netcdf_datetime(raw_time)
            if frame_time < start_time or frame_time > end_time:
                continue
            timestamp = frame_time.isoformat().replace("+00:00", "Z")
            filename = frame_time.strftime("%Y%m%dT%H%M%SZ.geojson")

            pressure = bilinear_resample(
                dataset.variables["msl"][index] / 100.0,
                source_latitudes,
                source_longitudes,
                target_latitudes,
                target_longitudes,
            )
            u_wind = bilinear_resample(
                dataset.variables["u10"][index],
                source_latitudes,
                source_longitudes,
                target_latitudes,
                target_longitudes,
            )
            v_wind = bilinear_resample(
                dataset.variables["v10"][index],
                source_latitudes,
                source_longitudes,
                target_latitudes,
                target_longitudes,
            )
            pressure = np.round(pressure, 1)
            u_wind = np.round(u_wind, 2)
            v_wind = np.round(v_wind, 2)
            wind_speed = np.hypot(u_wind, v_wind)

            pressure_min = min(pressure_min, float(np.min(pressure)))
            pressure_max = max(pressure_max, float(np.max(pressure)))
            wind_min = min(wind_min, float(np.min(wind_speed)))
            wind_max = max(wind_max, float(np.max(wind_speed)))
            pressure_url = f"pressure/{filename}"
            wind_url = f"wind/{filename}"
            _atomic_json(
                output_directory / pressure_url,
                _grid_geojson(
                    timestamp,
                    bounds,
                    target_latitudes,
                    target_longitudes,
                    "hPa",
                    {"pressure": pressure},
                ),
            )
            _atomic_json(
                output_directory / wind_url,
                _grid_geojson(
                    timestamp,
                    bounds,
                    target_latitudes,
                    target_longitudes,
                    "m/s",
                    {"u": u_wind, "v": v_wind},
                ),
            )
            frame_entries.append(
                {"timestamp": timestamp, "pressure": pressure_url, "wind": wind_url}
            )

    if not frame_entries:
        raise ValueError(f"no NetCDF frames fall inside {start_time.isoformat()}..{end_time.isoformat()}")

    manifest = {
        "schemaVersion": 1,
        "typhoon": typhoon,
        "name": info.get("name", typhoon.split("_", 1)[-1]),
        "bounds": bounds,
        "grid": {
            "width": int(target_longitudes.size),
            "height": int(target_latitudes.size),
            "resolution": resolution,
            "origin": "north-west",
            "order": "row-major",
        },
        "pressure": {
            "unit": "hPa",
            "min": round(float(pressure_min), 1),
            "max": round(float(pressure_max), 1),
        },
        "wind": {
            "unit": "m/s",
            "min": round(float(wind_min), 2),
            "max": round(float(wind_max), 2),
        },
        "frames": frame_entries,
    }
    _atomic_json(manifest_path, manifest)
    return ExportResult(typhoon, output_directory, len(frame_entries))

