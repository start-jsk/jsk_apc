#!/usr/bin/env python

from geometry_msgs.msg import Point, PointStamped


class BinData(object):
    def __init__(self, *args, **kwargs):
        if 'bin_info' in kwargs:
            self.from_bin_info(kwargs['bin_info'])

    def from_bin_info(self, bin_info, depth=1.0):

        self.bbox = bin_info.bbox
        self.in_blacklist = False
        self.target = bin_info.target
        self.bin_name = bin_info.name
        self.objects = bin_info.objects
        self.camera_direction = bin_info.camera_direction

        _header = self.bbox.header
        dimensions = [
            self.bbox.dimensions.x,
            self.bbox.dimensions.y,
            self.bbox.dimensions.z]
        initial_pos = [
            self.bbox.pose.position.x,
            self.bbox.pose.position.y,
            self.bbox.pose.position.z]
        assert type(dimensions[1]) == float
        for j in xrange(4):
            if self.camera_direction == 'x':
                # an order in which the points are appended is important
                # mask image depends on the order
                # x axis is directing away from Baxter
                self.corners = [
                        self.corner_point(
                                initial_pos, dimensions, depth, _header,
                                signs=[-1, 1, 1]),
                        self.corner_point(
                                initial_pos, dimensions, depth, _header,
                                signs=[-1, -1, 1]),
                        self.corner_point(
                                initial_pos, dimensions, depth, _header,
                                signs=[-1, -1, -1]),
                        self.corner_point(
                                initial_pos, dimensions, depth, _header,
                                signs=[-1, 1, -1])
                    ]
            elif self.camera_direction == 'y':
                raise NotImplementedError
            elif self.camera_direction == 'z':
                raise NotImplementedError
            else:
                raise NotImplementedError

    def corner_point(self, initial_pos, dimensions, depth, header, signs):
        return PointStamped(
                header=header,
                point=Point(
                        x=initial_pos[0]+signs[0]*depth*dimensions[0]/2,
                        y=initial_pos[1]+signs[1]*dimensions[1]/2,
                        z=initial_pos[2]+signs[2]*dimensions[2]/2))
