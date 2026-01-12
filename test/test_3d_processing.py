import numpy as np
import pytest

from src.depth_3d import generate_depth_map, generate_point_cloud_hw3, flatten_points

def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."

def test_generate_depth_map_none():
    with pytest.raises(ValueError):
        generate_depth_map(None)

def test_point_cloud_shapes_and_dtype():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    points_hw3 = generate_point_cloud_hw3(image)
    assert points_hw3.shape == (10, 20, 3)
    assert points_hw3.dtype == np.float32

    points_n3 = flatten_points(points_hw3)
    assert points_n3.shape == (10*20, 3)
    assert points_n3.dtype == np.float32
