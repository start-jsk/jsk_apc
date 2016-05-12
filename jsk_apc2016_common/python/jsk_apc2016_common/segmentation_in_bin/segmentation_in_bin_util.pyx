from cpython cimport array
cimport cython
from libc.math cimport abs

cdef double d2wall(double coord, double width):
    if coord >= 0 and coord < width/2:
        return abs(width/2 - coord)
    elif coord < 0 and abs(coord) < width/2:
        return abs(coord + width/2)
    else:
        return 0


cdef double d2front(double coord, double width):
    if abs(coord) <= width/2:
        return width/2 - coord
    else:
        return 0


cdef _get_spatial_feature(double point[3], double bbox_dimensions[3]):
    cdef double d2wall_x_back, d2wall_y, d2wall_z, d2wall_z_bottom

    d2wall_x_back = d2front(point[0], bbox_dimensions[0])
    d2wall_y = d2wall(point[1], bbox_dimensions[1])
    d2wall_z = d2wall(point[2], bbox_dimensions[2])
    d2wall_z_bottom = d2front(-point[2], bbox_dimensions[2])

    return (min(d2wall_x_back, d2wall_y, d2wall_z), d2wall_z_bottom)


def get_spatial_feature(points, bbox_dimensions):
    cdef array.array points_array = array.array('d', points)
    cdef array.array bbox_dimensions_array = array.array('d', bbox_dimensions)
    return _get_spatial_feature(points_array.data.as_doubles,
                                bbox_dimensions_array.data.as_doubles)
