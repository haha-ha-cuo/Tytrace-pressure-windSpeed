from __future__ import annotations

import numpy as np
import numpy.typing as npt


def build_axis(start: float, stop: float, resolution: float) -> npt.NDArray[np.float64]:
    """Build an inclusive axis without extending beyond the source bounds."""
    if not np.isfinite([start, stop, resolution]).all():
        raise ValueError("axis bounds and resolution must be finite")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if stop < start:
        raise ValueError("axis stop must not be smaller than start")
    if np.isclose(start, stop):
        return np.asarray([start], dtype=np.float64)

    count = int(np.floor((stop - start) / resolution + 1e-10))
    axis = start + np.arange(count + 1, dtype=np.float64) * resolution
    axis = axis[axis <= stop + 1e-10]
    if not np.isclose(axis[-1], stop):
        axis = np.append(axis, stop)
    else:
        axis[-1] = stop
    return axis


def _validate_axis(name: str, axis: npt.NDArray[np.float64]) -> int:
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two values")
    differences = np.diff(axis)
    if np.all(differences > 0):
        return 1
    if np.all(differences < 0):
        return -1
    raise ValueError(f"{name} must be strictly monotonic")


def bilinear_resample(
    values: npt.ArrayLike,
    source_latitudes: npt.ArrayLike,
    source_longitudes: npt.ArrayLike,
    target_latitudes: npt.ArrayLike,
    target_longitudes: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Resample a latitude/longitude field with no extrapolation."""
    field = np.ma.asarray(values, dtype=np.float64)
    source_lat = np.asarray(source_latitudes, dtype=np.float64)
    source_lon = np.asarray(source_longitudes, dtype=np.float64)
    target_lat = np.asarray(target_latitudes, dtype=np.float64)
    target_lon = np.asarray(target_longitudes, dtype=np.float64)

    lat_direction = _validate_axis("source_latitudes", source_lat)
    lon_direction = _validate_axis("source_longitudes", source_lon)
    if field.shape != (source_lat.size, source_lon.size):
        raise ValueError("values shape must match the latitude and longitude axes")
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("target axes must be one-dimensional")
    if target_lat.size == 0 or target_lon.size == 0:
        raise ValueError("target axes must not be empty")
    if np.ma.is_masked(field) and np.ma.getmaskarray(field).any():
        raise ValueError("source field contains missing values")

    if lat_direction < 0:
        source_lat = source_lat[::-1]
        field = field[::-1, :]
    if lon_direction < 0:
        source_lon = source_lon[::-1]
        field = field[:, ::-1]

    epsilon = 1e-9
    if (
        target_lat.min() < source_lat[0] - epsilon
        or target_lat.max() > source_lat[-1] + epsilon
        or target_lon.min() < source_lon[0] - epsilon
        or target_lon.max() > source_lon[-1] + epsilon
    ):
        raise ValueError("target coordinates fall outside the source grid")

    lat1 = np.clip(np.searchsorted(source_lat, target_lat, side="right"), 1, source_lat.size - 1)
    lon1 = np.clip(np.searchsorted(source_lon, target_lon, side="right"), 1, source_lon.size - 1)
    lat0 = lat1 - 1
    lon0 = lon1 - 1
    lat_weight = (target_lat - source_lat[lat0]) / (source_lat[lat1] - source_lat[lat0])
    lon_weight = (target_lon - source_lon[lon0]) / (source_lon[lon1] - source_lon[lon0])

    north_west = np.asarray(field[np.ix_(lat0, lon0)])
    north_east = np.asarray(field[np.ix_(lat0, lon1)])
    south_west = np.asarray(field[np.ix_(lat1, lon0)])
    south_east = np.asarray(field[np.ix_(lat1, lon1)])
    horizontal_low = north_west * (1 - lon_weight) + north_east * lon_weight
    horizontal_high = south_west * (1 - lon_weight) + south_east * lon_weight
    return horizontal_low * (1 - lat_weight[:, None]) + horizontal_high * lat_weight[:, None]

