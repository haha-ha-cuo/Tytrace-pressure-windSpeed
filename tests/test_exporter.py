import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from tytrace_fields.exporter import export_typhoon


def create_dataset(root: Path) -> None:
    directory = root / "2501_TEST"
    directory.mkdir(parents=True)
    (directory / "info.json").write_text(
        json.dumps(
            {
                "name": "TEST",
                "start_time": "2025-01-01 01:00",
                "end_time": "2025-01-01 02:00",
            }
        ),
        encoding="utf-8",
    )
    with Dataset(directory / "2501_TEST.nc", "w") as dataset:
        dataset.createDimension("valid_time", 3)
        dataset.createDimension("latitude", 2)
        dataset.createDimension("longitude", 2)
        times = dataset.createVariable("valid_time", "i8", ("valid_time",))
        times.units = "seconds since 1970-01-01"
        times[:] = [1735689600, 1735693200, 1735696800]
        dataset.createVariable("latitude", "f8", ("latitude",))[:] = [1, 0]
        dataset.createVariable("longitude", "f8", ("longitude",))[:] = [100, 101]
        pressure = dataset.createVariable("msl", "f4", ("valid_time", "latitude", "longitude"))
        u_wind = dataset.createVariable("u10", "f4", ("valid_time", "latitude", "longitude"))
        v_wind = dataset.createVariable("v10", "f4", ("valid_time", "latitude", "longitude"))
        pressure_frame = np.asarray([[100000, 100100], [100200, 100300]])
        for index in range(3):
            pressure[index, :, :] = pressure_frame
            u_wind[index, :, :] = np.full((2, 2), 3)
            v_wind[index, :, :] = np.full((2, 2), 4)


def test_exporter_crops_time_converts_units_and_writes_geojson(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "weatherdata"
    create_dataset(dataset_root)

    result = export_typhoon(dataset_root, output_root, "2501_TEST", resolution=0.5)

    assert result.frame_count == 2
    manifest = json.loads((result.output_directory / "manifest.json").read_text())
    assert manifest["bounds"] == [100.0, 0.0, 101.0, 1.0]
    assert manifest["grid"] == {
        "width": 3,
        "height": 3,
        "resolution": 0.5,
        "origin": "north-west",
        "order": "row-major",
    }
    assert manifest["pressure"] == {"unit": "hPa", "min": 1000.0, "max": 1003.0}
    assert manifest["wind"] == {"unit": "m/s", "min": 5.0, "max": 5.0}
    pressure_path = result.output_directory / manifest["frames"][0]["pressure"]
    pressure = json.loads(pressure_path.read_text())
    assert pressure["type"] == "FeatureCollection"
    assert pressure["features"] == []
    assert pressure["grid"]["pressure"][4] == 1001.5
