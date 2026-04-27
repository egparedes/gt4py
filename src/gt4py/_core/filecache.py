# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import pathlib
import pickle
from typing import Any, Hashable, Generic, TypeVar, TYPE_CHECKING

from gt4py._core import locking
from gt4py.eve import utils as eve_utils

T = TypeVar("T")


class FileCache(Generic[T]):
    """
    Dictionary-like object for persistently caching objects on disk.
    """

    path: pathlib.Path

    def __init__(self, path: str | os.PathLike):
        self.path = pathlib.Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: Hashable) -> pathlib.Path:
        """Return the path where an item with `key` is stored."""
        key = eve_utils.content_hash(key)
        path = self.path / f"{key}.pkl"
        return path

    def __getitem__(self, key: Hashable) -> T:
        if key not in self:
            raise KeyError(key)
        with locking.lock(path := self._get_path(key)):
            with open(path, "rb") as f:
                return pickle.load(f)

    def __setitem__(self, key: Hashable, value: T) -> None:
        with locking.lock(path := self._get_path(key)):
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=5)

    def __delitem__(self, key: Hashable) -> None:
        if key not in self:
            raise KeyError(key)
        with locking.lock(path := self._get_path(key)):
            path.unlink()

    def __contains__(self, key: Hashable) -> bool:
        return self._get_path(key).exists()


if TYPE_CHECKING:
    from gt4py.eve.extended_typing import OpaqueMutableMapping

    _FC: type[OpaqueMutableMapping[Hashable, int]] = FileCache[int]
