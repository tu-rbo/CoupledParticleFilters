"""Small ROS2 bag reader and image decoder used by the offline adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CorruptRosbagError(RuntimeError):
    """A bag cannot be read because its metadata, database, or payload is damaged."""


class ROS2BagReader:
    def __init__(self, bagfile: str | Path) -> None:
        from rosbags.rosbag2 import Reader

        path = Path(bagfile)
        self.bag_path = path.parent if path.suffix == ".db3" else path
        self.reader = Reader(self.bag_path)
        try:
            self.reader.open()
        except Exception as exc:
            if is_corrupt_bag_exception(exc):
                raise CorruptRosbagError(f"cannot open converted bag: {exc}") from exc
            raise

    def read_messages(self, topics: list[str]):
        topic_set = set(topics)
        try:
            for connection, timestamp, rawdata in self.reader.messages():
                if not topic_set or connection.topic in topic_set:
                    yield connection.topic, rawdata, timestamp
        except Exception as exc:
            if is_corrupt_bag_exception(exc):
                raise CorruptRosbagError(f"cannot read converted bag database: {exc}") from exc
            raise

    def close(self) -> None:
        self.reader.close()


class RGBDImageDecoder:
    def imgmsg_to_cv2(self, message: Any) -> torch.Tensor:
        dtype = np.dtype("uint8").newbyteorder(">" if message.is_bigendian else "<")
        image = np.ndarray(
            shape=(message.height, message.width, 3),
            dtype=dtype,
            buffer=message.data,
        )
        if message.is_bigendian == (os.sys.byteorder == "little"):
            image = image.byteswap().newbyteorder()
        return torch.as_tensor(np.array(image))

    def depthmsg_to_cv2(self, message: Any) -> torch.Tensor:
        dtype = np.dtype("uint16") if message.encoding == "16UC1" else np.dtype("float32")
        dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
        image = np.ndarray(
            shape=(message.height, message.width),
            dtype=dtype,
            buffer=message.data,
        )
        if message.is_bigendian == (os.sys.byteorder == "little"):
            image = image.byteswap().newbyteorder()
        depth = np.array(image).astype(np.float32)
        if message.encoding == "16UC1":
            depth /= 1000.0
        return torch.as_tensor(depth)


def is_corrupt_bag_exception(exc: BaseException) -> bool:
    """Conservatively recognize storage/metadata failures, not arbitrary pipeline bugs."""
    try:
        from rosbags.rosbag2 import ReaderError
    except ImportError:
        ReaderError = ()  # type: ignore[assignment,misc]

    if ReaderError and isinstance(exc, ReaderError):
        return True
    message = str(exc).lower()
    corruption_markers = (
        "database disk image is malformed",
        "file is not a database",
        "database malformed",
        "malformed database schema",
        "could not read metadata",
        "metadata.yaml",
        "no storage file found",
        "not a rosbag",
    )
    return any(marker in message for marker in corruption_markers)
