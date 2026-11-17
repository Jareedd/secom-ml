"""Load the raw UCI SECOM files into time-ordered tables."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from secom_ml.config import DATA_DIR

SECOM_ZIP_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SENSOR_FILE = "secom.data"
LABEL_FILE = "secom_labels.data"

# Timestamps are written day/month/year. The format is spelled out so pandas
# never guesses: a guess could read 01/08/2008 as January 8 and silently
# scramble the time order that the train/test split depends on.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


def download_if_missing(data_dir: Path = DATA_DIR) -> None:
    """Fetch the two raw files from UCI unless they are already on disk."""
    if (data_dir / SENSOR_FILE).exists() and (data_dir / LABEL_FILE).exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"data/ is missing, downloading {SECOM_ZIP_URL}")
    with urllib.request.urlopen(SECOM_ZIP_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in (SENSOR_FILE, LABEL_FILE):
        (data_dir / name).write_bytes(archive.read(name))


def read_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both files exactly as written, in file order (no sorting).

    Returns (sensors, labels). labels has the columns "label" (-1 pass, 1 fail)
    and "timestamp".
    """
    download_if_missing(data_dir)

    sensors = pd.read_csv(data_dir / SENSOR_FILE, sep=" ", header=None)
    sensors.columns = [f"s{i:03d}" for i in range(sensors.shape[1])]

    labels = pd.read_csv(
        data_dir / LABEL_FILE, sep=" ", header=None, names=["label", "timestamp"]
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format=TIMESTAMP_FORMAT)

    if len(sensors) != len(labels):
        raise ValueError(f"{len(sensors)} sensor rows but {len(labels)} label rows")
    if not set(labels["label"].unique()) <= {-1, 1}:
        raise ValueError(f"unexpected label values: {labels['label'].unique()}")
    return sensors, labels


def load_secom(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, timestamp) with rows sorted oldest to newest.

    X: one row per production run, one column per sensor (s000, s001, ...).
    y: 1 if the run failed, 0 if it passed.
    timestamp: when the run happened.

    The sort is stable, so runs that share a timestamp keep their file order
    and the result is the same on every machine.
    """
    sensors, labels = read_raw(data_dir)
    order = labels["timestamp"].sort_values(kind="stable").index

    X = sensors.loc[order].reset_index(drop=True)
    y = (labels.loc[order, "label"] == 1).astype(int).rename("fail").reset_index(drop=True)
    timestamp = labels.loc[order, "timestamp"].reset_index(drop=True)
    return X, y, timestamp
