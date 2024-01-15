import copy
import json
import math
import os
import re
import struct
import subprocess
from random import random
from typing import Dict, Optional, List, Union, Tuple, Any
import tqdm
from enum import Enum

min_x, min_z, min_y = math.inf, math.inf, math.inf
max_x, max_z, max_y = -math.inf, -math.inf, -math.inf


class FileManager:
    def __init__(self):
        self.files = {}
        self.files_meta = {}

    def save_to_disk(self, filename: str, content):
        if filename not in self.files:
            file = open(filename, 'w')
            self.files[filename] = file
        self.files[filename].write(content)

    def close(self):
        for file in self.files.values():
            file.close()

    def flush(self):
        for file in self.files.values():
            file.flush()


file_manager: Optional[FileManager] = None


class PCDHeaders:
    def __init__(self):
        # PCD header fields
        self.version = "0.7"
        self.fields = []
        self.size = []
        self.type = []
        self.count = []
        self.width = 0
        self.height = 0
        self.viewpoint = []
        self.points: Union[int, List[str], None] = None
        self.data = ""
        self.header_len = None
        self.str = ""
        self.offset = {}
        self.row_size = 0

    def display_header(self):
        print("PCD Header:")
        print("Version:", self.version)
        print("Fields:", self.fields)
        print("Size:", self.size)
        print("Type:", self.type)
        print("Count:", self.count)
        print("Width:", self.width)
        print("Height:", self.height)
        print("Viewpoint:", self.viewpoint)
        print("Points:", self.points)
        print("Data:", self.data)


point_cloud_sequential_color_array = [
    '#440154',
    '#470e61',
    '#481b6d',
    '#482576',
    '#46307e',
    '#443b84',
    '#404688',
    '#3c508b',
    '#38598c',
    '#33628d',
    '#2f6b8e',
    '#2c738e',
    '#287c8e',
    '#25838e',
    '#228c8d',
    '#1f948c',
    '#1e9d89',
    '#20a486',
    '#26ad81',
    '#31b57b',
    '#3fbc73',
    '#50c46a',
    '#60ca60',
    '#75d054',
    '#8bd646',
    '#a2da37',
    '#bade28',
    '#d0e11c',
    '#e7e419',
    '#fde725',
]


class FragmentMeta:
    def __init__(self, count: int, levels: List[str]):
        self.count = count
        self.levels = levels

    def as_json(self):
        return {
            'count': self.count,
            'levels': self.levels,
        }


class IndexedPointCloudConfig:
    def __init__(
            self,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            min_z: float,
            max_z: float,
            cols: int,
            rows: int,
            row_size: int,
            col_size: int,
            stack_size: int,
            stacks: int,
            keys: Dict[str, FragmentMeta],
            total_points: int,
            id: Optional[str] = None
    ):
        # Initialize attributes with values
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z
        self.cols = cols
        self.rows = rows
        self.row_size = row_size
        self.col_size = col_size
        self.stack_size = stack_size
        self.stacks = stacks
        self.keys = keys
        self.total_points = total_points
        self.id = id

    def as_json(self):
        keys_dict = {}
        for key in self.keys:
            keys_dict[key] = self.keys[key].as_json()
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "cols": self.cols,
            "rows": self.rows,
            "row_size": self.row_size,
            "col_size": self.col_size,
            "stack_size": self.stack_size,
            "stacks": self.stacks,
            "keys": keys_dict,
            "total_points": self.total_points
        }


class PointAddressType:
    def __init__(self, point_3d_index: int, fragment_index: int, fragment_key: str, fragment_number: int):
        self.point_3d_index = point_3d_index
        self.fragment_index = fragment_index
        self.fragment_key = fragment_key
        self.fragment_number = fragment_number


class GeometryAttributesData:
    def __init__(
            self,
            position: Union[List[float], List[int]],
            pcd_color: Union[List[float], List[int]],
            original_color: Union[List[float], List[int]],
            intensity_color: Optional[List[float]] = None,
            velocity_color: Optional[List[float]] = None,
            original_velocity_values: Optional[List[float]] = None,
            color_mapping: Optional[List[float]] = None,
            original_intensity_values: Optional[List[float]] = None,
            label_key: Optional[List[int]] = None
    ):
        self.position = position
        self.pcd_color = pcd_color
        self.original_color = original_color
        self.intensity_color = intensity_color
        self.velocity_color = velocity_color
        self.original_velocity_values = original_velocity_values
        self.color_mapping = color_mapping
        self.original_intensity_values = original_intensity_values
        self.label_key = label_key

    def as_json(self):
        return self.__dict__


class IndexedGeometryAttributesData(GeometryAttributesData):
    def __init__(self,
                 indices: List[int],
                 apc_addresses: List[PointAddressType],
                 count: int,
                 position: Union[List[float], List[int]],
                 pcd_color: Union[List[float], List[int]],
                 original_color: Union[List[float], List[int]],
                 intensity_color: Optional[List[float]] = None,
                 velocity_color: Optional[List[float]] = None,
                 original_velocity_values: Optional[List[int]] = None,
                 color_mapping: Optional[List[float]] = None,
                 original_intensity_values: Optional[List[float]] = None,
                 label_key: Optional[List[int]] = None):
        super().__init__(position, pcd_color, original_color, intensity_color, velocity_color, original_velocity_values,
                         color_mapping, original_intensity_values, label_key)
        self.indices = indices
        self.apc_addresses = apc_addresses
        self.count = count

    def as_dict(self):
        super_dict = super().as_json()
        super_dict['apc_addresses'] = self.apc_addresses
        super_dict['count'] = self.count
        super_dict['indices'] = self.indices
        return super_dict


class LevelledIndexedGeometryAttributesData:
    def __init__(self, levels: Dict[str, IndexedGeometryAttributesData], count: int):
        self.levels = levels
        self.count = count

    def as_dict(self):
        return {
            'count': self.count,
            'levels': self.levels.__dict__
        }


class SpatialData:
    def __init__(self, config: Union[IndexedPointCloudConfig, None] = None,
                 fragments: Dict[str, LevelledIndexedGeometryAttributesData] = None):
        self.config = config
        self.fragments = fragments
        if fragments is None:
            self.fragments = {}


class PointAttributes:
    def __init__(self):
        self.color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.intensity_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.color_mapping: Union[None, Tuple[float, float, float]] = None
        self.original_intensity: Union[None, float] = None
        self.original_velocity: Union[None, float] = None
        self.velocity_color: Union[None, Tuple[float, float, float]] = None
        self.label_key: Union[None, int] = None
        self.index: Union[None, int] = None
        self.apc_address: Union[None, PointAddressType] = None


def parse_pcd_headers(data: str) -> PCDHeaders:
    # Search for a pattern in the 'data' string using a regular expression
    result1 = re.search(r'[\r\n]DATA\s(\S*)\s', data, re.I)

    # Extract information from the substring of 'data' starting from result1.end() - 1
    result2 = result1.groups()[-1]

    if result2 is None:
        raise Exception('PCD data is undefined')

    headers = PCDHeaders()
    headers.data = result2
    headers.header_len = result1.end()
    headers.str = data[:headers.header_len]

    # parse headers
    headers.version = re.search(r"VERSION (.*)", headers.str, re.IGNORECASE).group(1)
    headers.fields = re.search(r"FIELDS (.*)", headers.str, re.IGNORECASE).group(1).split()
    headers.size = re.search(r"SIZE (.*)", headers.str, re.IGNORECASE).group(1).split()
    headers.type = re.search(r"TYPE (.*)", headers.str, re.IGNORECASE).group(1).split()
    headers.count = re.search(r"COUNT (.*)", headers.str, re.IGNORECASE).group(1).split()
    headers.width = re.search(r"WIDTH (.*)", headers.str, re.IGNORECASE).group(1)
    headers.height = re.search(r"HEIGHT (.*)", headers.str, re.IGNORECASE).group(1)
    headers.viewpoint = re.search(r"VIEWPOINT (.*)", headers.str, re.IGNORECASE)
    headers.points = re.search(r"POINTS (.*)", headers.str, re.IGNORECASE).group(1)

    if headers.viewpoint is not None:
        headers.viewpoint = headers.viewpoint.group(1)

    # convert to proper types
    headers.version = float(headers.version)
    headers.size = list(map(int, headers.size))
    headers.count = list(map(int, headers.count))
    headers.width = int(headers.width)
    headers.height = int(headers.height)
    headers.points = int(headers.points)

    size_sum = 0
    i = 0
    l = len(headers.fields)

    while i < l:
        if headers.data.lower().strip() == 'ascii':
            headers.offset[headers.fields[i]] = i
        # binary and binary_compressed are ignored

        i += 1

    headers.row_size = size_sum
    return headers


def get_total_lines(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    total_lines = int(result.stdout.split()[0])
    return total_lines


def main(file_name):
    global file_manager
    try:
        file_manager = FileManager()
        parse_point_cloud(file_name)
    finally:
        file_manager.close()


def parse_point_cloud(file_name):
    global min_x, max_x, min_y, max_y, min_z, max_z
    headers = PCDHeaders()
    string_data = ""

    row_size, column_size, stack_size = 0.2, 0.2, 0.2
    if file_name == '15M_pcd.pcd':
        row_size, column_size, stack_size = 10, 10, 5

    levels = [1, 2, 5]
    spatial_data: SpatialData = SpatialData(config=None, fragments={})
    fragments_meta = {}

    pbar = None
    i = 0
    random_number = math.floor(random() * 999999999)
    generated_data_dir = '../generated-data'
    random_dir_for_intermediate_pcd_files = generated_data_dir + '/intermediate_pcd_files_' + str(random_number)
    all_fragment_keys = set()
    with open(file_name, 'r') as original_pcd_file:
        for line in original_pcd_file:
            if headers.points is None:
                string_data += line + '\n'
                if line.find('DATA') != -1:
                    headers = parse_pcd_headers(string_data)
                    print(f'File {file_name}\nTotal Points: {headers.points}\nFields: {", ".join(headers.fields)}')
                    print('Fragmenting the PCD File...')
                    pbar = tqdm.tqdm(total=headers.points, unit=' line')
            else:
                pbar.update(1)
                if line == '':
                    continue

                point_attributes = parse_point(line, headers, i)

                px, py, pz = point_attributes.position
                fragment_x = math.floor(px / column_size)
                fragment_y = math.floor(py / row_size)
                fragment_z = math.floor(pz / stack_size)

                fragment_key = f"{fragment_x}_{fragment_y}_{fragment_z}"
                all_fragment_keys.add(fragment_key)
                base_level = levels[0]
                applicable_levels = get_levels_that_cover_this_point(fragments_meta, levels, fragment_key, base_level)
                if len(applicable_levels) > 0:
                    save_point_to_file(random_dir_for_intermediate_pcd_files, fragment_key, applicable_levels, line, i)
            i += 1

    if pbar is not None:
        pbar.close()
    file_manager.flush()

    # now that we have assigned levels to points and divided in fragments, we'll go through all the fragment files and
    # convert those to required json format
    # fetch all files
    all_paths_in_intermediate_dir = os.listdir(random_dir_for_intermediate_pcd_files)
    files_in_intermediate_dir = []
    for path in all_paths_in_intermediate_dir:
        full_path = random_dir_for_intermediate_pcd_files + '/' + path
        if os.path.isfile(full_path):
            files_in_intermediate_dir.append(full_path)

    min_fragment_x = math.inf
    min_fragment_y = math.inf
    min_fragment_z = math.inf
    for key in all_fragment_keys:
        fragment_x, fragment_y, fragment_z = map(int, key.split('_'))
        min_fragment_x = fragment_x if fragment_x < min_fragment_x else min_fragment_x
        min_fragment_y = fragment_y if fragment_y < min_fragment_y else min_fragment_y
        min_fragment_z = fragment_z if fragment_z < min_fragment_z else min_fragment_z

    new_keys = {}
    dir_for_final_files = f"{generated_data_dir}/final_pcd_files_{random_number}"
    for file in tqdm.tqdm(files_in_intermediate_dir):
        fragments: Dict[str, LevelledIndexedGeometryAttributesData] = {}

        with open(file) as f:
            for line in f.readlines():
                line_arr = line.split(' ', 3)
                fragment_key = line_arr[0]
                point_levels = list(map(int, line_arr[1].split(',')))
                point_index = int(line_arr[2])
                point_data = line_arr[3]

                if fragment_key not in fragments:
                    fragments[fragment_key] = LevelledIndexedGeometryAttributesData(levels={}, count=0)
                fragments[fragment_key].count += 1

                for point_level in point_levels:
                    current_level = str(point_level)
                    if str(current_level) not in fragments[fragment_key].levels:
                        fragments[fragment_key].levels[str(current_level)] = IndexedGeometryAttributesData(
                            indices=[],
                            apc_addresses=[],
                            count=0,
                            position=[],
                            pcd_color=[],
                            original_color=[],
                            intensity_color=[],
                            velocity_color=[],
                            original_velocity_values=[],
                            color_mapping=[],
                            original_intensity_values=[],
                            label_key=[]
                        )

                    point_attrs = parse_point(point_data, headers, point_index)

                    fragments[fragment_key].levels[current_level].count += 1
                    if point_attrs.index is not None:
                        fragments[fragment_key].levels[current_level].indices.append(point_attrs.index)
                    if point_attrs.position is not None:
                        fragments[fragment_key].levels[current_level].position.extend(point_attrs.position)
                    if point_attrs.color is not None:
                        fragments[fragment_key].levels[current_level].pcd_color.extend(point_attrs.color)
                    if point_attrs.intensity_color is not None:
                        fragments[fragment_key].levels[current_level].intensity_color.extend(
                            point_attrs.intensity_color)
                    if point_attrs.original_intensity is not None:
                        fragments[fragment_key].levels[current_level].original_intensity_values.append(
                            point_attrs.original_intensity)
                    if point_attrs.color_mapping is not None:
                        fragments[fragment_key].levels[current_level].color_mapping.extend(point_attrs.color_mapping)
                    if point_attrs.velocity_color is not None:
                        fragments[fragment_key].levels[current_level].velocity_color.extend(point_attrs.velocity_color)
                    if point_attrs.original_velocity is not None:
                        fragments[fragment_key].levels[current_level].original_velocity_values.append(
                            point_attrs.original_velocity)
                    if point_attrs.label_key is not None:
                        fragments[fragment_key].levels[current_level].label_key.append(point_attrs.label_key)
                    if point_attrs.apc_address is not None:
                        fragments[fragment_key].levels[current_level].apc_addresses.append(point_attrs.apc_address)

        fragment_keys = fragments.keys()
        for key in copy.deepcopy(list(fragment_keys)):
            fragment = fragments[key]
            fragment_x, fragment_y, fragment_z = map(int, key.split('_'))
            normalized_fragment_x = fragment_x - min_fragment_x
            normalized_fragment_y = fragment_y - min_fragment_y
            normalized_fragment_z = fragment_z - min_fragment_z
            new_key = f"{normalized_fragment_x}-{normalized_fragment_y}-{normalized_fragment_z}"

            fragments[new_key] = fragment
            del fragments[key]
            levels = fragment.levels
            new_keys[new_key] = FragmentMeta(
                count=fragment.count,
                levels=copy.deepcopy(list(levels.keys()))
            )

            dir_path = f"{dir_for_final_files}/{new_key}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for level_key in copy.deepcopy(levels):
                level_data = fragment.levels[level_key]
                level_save_path = f"{dir_path}/{level_key}.json"
                with open(level_save_path, 'w') as f:
                    f.write(json.dumps(level_data.as_dict()))
                    del fragment.levels[level_key]

    print(f"saved to {dir_for_final_files}")
    spatial_data.config = IndexedPointCloudConfig(
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
        0,
        0,
        row_size,
        column_size,
        stack_size,
        0,
        new_keys,
        total_points=i,
    )
    with open(f"{dir_for_final_files}/config.json", 'w') as f:
        f.write(json.dumps(spatial_data.config.as_json()))


def parse_point(raw_point, headers, i):
    global min_x, max_x, min_y, max_y, min_z, max_z
    line_arr = raw_point.split()

    point_attributes: PointAttributes = PointAttributes()
    if 'x' in headers.offset:
        x = float(line_arr[headers.offset['x']])
        y = float(line_arr[headers.offset['y']])
        z = float(line_arr[headers.offset['z']])

        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            min_x = x if x < min_x else min_x
            min_y = y if y < min_y else min_y
            min_z = z if z < min_z else min_z

            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y
            max_z = z if z > max_z else max_z
            point_attributes.index = i
            point_attributes.position = [x, y, z]

    if 'v' in headers.offset:
        # todo: add code for velocity color
        pass

    if 'label' in headers.offset:
        # todo: add code for label
        pass

    if 'label_key' in headers.offset:
        # todo: add code for label_key
        pass

    if 'rgb' in headers.offset:
        # todo: fix inaccuracy in packing, unpacking
        packed_data = struct.pack('f', float(line_arr[headers.offset['rgb']]))
        unpacked_data = struct.unpack('bbbb', packed_data)
        # print(line_arr[headers.offset['rgb']], float(line_arr[headers.offset['rgb']]), packed_data, unpacked_data)
        # raise Exception('stop')
        r, g, b, _ = unpacked_data
        point_attributes.color = (
            round((r / 255) * 10) / 10,
            round((g / 255) * 10) / 10,
            round((b / 255) * 10) / 10,
        )
    else:
        point_attributes.color = (1, 1, 1)

    if 'intensity' in headers.offset or 'reflectivity' in headers.offset:
        # Create and push alpha wrt intensity
        alpha_value = float(headers.offset.get('intensity', headers.offset.get('reflectivity')))
        alpha_value = alpha_value / 255 if alpha_value > 1 else alpha_value
        point_attributes.intensity_color = check_for_valid_alpha_value_and_add_intensity_color(alpha_value)
        point_attributes.original_intensity = float(line_arr[headers.offset['intensity']])

    if 'r' in headers.offset and 'g' in headers.offset and 'b' in headers.offset:
        r = float(line_arr[headers.offset['r']]) / 255
        g = float(line_arr[headers.offset['g']]) / 255
        b = float(line_arr[headers.offset['b']]) / 255
        point_attributes.color_mapping = (r, g, b)

    return point_attributes


def get_levels_that_cover_this_point(
        fragments: dict,
        levels_to_apply: List[int],
        fragment_key: str,
        base_level: int
):
    selected_levels = []
    base_level_fragment_key = f"{fragment_key}_{base_level}"
    if base_level_fragment_key not in fragments:
        for level in levels_to_apply:
            fragments[f"{fragment_key}_{level}"] = {'count': 0}
    for level in levels_to_apply:
        current_level_fragment_key = f"{fragment_key}_{level}"

        # if level is n, then only select every nth point in this fragment
        # i.e. only if count in base level % curr_level == 0
        if fragments[base_level_fragment_key]['count'] % level != 0:
            continue
        selected_levels.append(level)
        fragments[current_level_fragment_key]['count'] = fragments[current_level_fragment_key]['count'] + 1

    return selected_levels


def save_point_to_file(directory, fragment_key, levels, point, point_index):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = f"{directory}/{fragment_key}.pcd"
    data_to_save = fragment_key + " " + ",".join([str(level) for level in levels]) + " " + str(
        point_index) + " " + point
    file_manager.save_to_disk(file_path, data_to_save)


def check_for_valid_alpha_value_and_add_intensity_color(alpha_value):
    """
    If alpha value lies between 0 -> 1 then only consider it as valid intensity value
    Intensity value are allowed only if lies in the range of 0-1 else show point colors
    toggle wont be shown
    """
    if not math.isnan(alpha_value) and alpha_value <= 1:
        return list(map((lambda color: color / 255), get_color_wrt_intensity(alpha_value)))

    return []


def get_color_wrt_intensity(intensity):
    return hex_to_rgb(
        point_cloud_sequential_color_array[
            math.floor((len(point_cloud_sequential_color_array) - 1) * intensity)
        ]
    )


def hex_to_rgb(color) -> Union[Tuple[int, int, int], str]:
    if color.find('rgb') == -1:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return r, g, b
    return color


class PCDVelocityColorConfig(Enum):
    PositiveRGB = (81, 172, 248)
    NeutralRGB = (255, 255, 255)
    NegativeRGB = (117, 222, 133)


def get_velocity_color_wrt_velocity_value(velocity):
    rgb = PCDVelocityColorConfig.NeutralRGB
    original_velocity_value = 0
    if velocity > 0:
        rgb = PCDVelocityColorConfig.PositiveRGB
        original_velocity_value = 1
    elif velocity < 0:
        rgb = PCDVelocityColorConfig.NegativeRGB
        original_velocity_value = -1

    # todo: finish this function
