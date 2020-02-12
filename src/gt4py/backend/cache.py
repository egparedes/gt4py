# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pickle
import sys
import types

from gt4py import config as gt_config
from gt4py import utils as gt_utils
from gt4py import __version__ as gt_version


def get_stencil_class_name(stencil_id):
    components = stencil_id.qualified_name.split(".")
    class_name = "{name}__{id}".format(name=components[-1], id=stencil_id.version)
    return class_name


def get_stencil_package_name(stencil_id):
    components = stencil_id.qualified_name.split(".")
    package_name = ".".join([gt_config.code_settings["root_package_name"]] + components[:-1])
    return package_name


def get_stencil_module_name(stencil_id, *, qualified=False):
    module_name = "m_{}".format(get_stencil_class_name(stencil_id))
    if qualified:
        module_name = "{}.{}".format(get_stencil_package_name(stencil_id), module_name)
    return module_name


def get_base_path(backend_id):
    # Initialize cache folder
    cache_root = os.path.join(
        gt_config.cache_settings["root_path"], gt_config.cache_settings["dir_name"]
    )
    if not os.path.exists(cache_root):
        gt_utils.make_dir(cache_root, is_cache=True)

    cpython_id = "py{major}{minor}_{api}".format(
        major=sys.version_info.major, minor=sys.version_info.minor, api=sys.api_version
    )
    base_path = os.path.join(cache_root, cpython_id, gt_utils.slugify(backend_id))
    return base_path


def get_stencil_package_path(backend_id, stencil_id):
    components = stencil_id.qualified_name.split(".")
    path = os.path.join(get_base_path(backend_id), *components[:-1])
    return path


def get_stencil_module_path(backend_id, stencil_id):
    stencil_module_name = get_stencil_module_name(stencil_id)
    path = os.path.join(
        get_stencil_package_path(backend_id, stencil_id), stencil_module_name + ".py"
    )
    return path


def get_cache_info_path(backend_id, stencil_id):
    path = str(get_stencil_module_path(backend_id, stencil_id))[:-3] + ".cacheinfo"
    return path


def generate_cache_info(backend_id, stencil_id, extra_cache_info):
    module_file_name = get_stencil_module_path(backend_id, stencil_id)
    with open(module_file_name, "r") as f:
        source = f.read()
    cache_info = {
        "gt4py_version": gt_version,
        "backend": backend_id,
        "stencil_name": stencil_id.qualified_name,
        "stencil_version": stencil_id.version,
        "module_shash": gt_utils.shash(source),
        **extra_cache_info,
    }

    return cache_info


def update_cache(backend_id, stencil_id, extra_cache_info):
    cache_info = generate_cache_info(backend_id, stencil_id, extra_cache_info)
    cache_file_name = get_cache_info_path(backend_id, stencil_id)
    os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
    with open(cache_file_name, "wb") as f:
        pickle.dump(cache_info, f)


def validate_cache_info(backend_id, stencil_id, cache_info):
    try:
        cache_info = types.SimpleNamespace(**cache_info)

        module_file_name = get_stencil_module_path(backend_id, stencil_id)
        with open(module_file_name, "r") as f:
            source = f.read()
        module_shash = gt_utils.shash(source)

        result = (
            cache_info.backend == backend_id
            and cache_info.stencil_name == stencil_id.qualified_name
            and cache_info.stencil_version == stencil_id.version
            and cache_info.module_shash == module_shash
        )

    except Exception:
        result = False

    return result


def check_cache(backend_id, stencil_id):
    try:
        cache_file_name = get_cache_info_path(backend_id, stencil_id)
        with open(cache_file_name, "rb") as f:
            cache_info = pickle.load(f)
        assert isinstance(cache_info, dict)
        result = validate_cache_info(backend_id, stencil_id, cache_info)

    except Exception:
        result = False

    return result
