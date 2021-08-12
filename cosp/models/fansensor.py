import math
import random
import numpy as np

from .sensor import SensorModel
from ..utils.math import to_rad, R2d, euclidean_dist, pol2cart
from ..utils.misc import in_range_inclusive

class FanSensor(SensorModel):
    def __init__(self, name="laser2d_sensor", **params):
        """
        2D fanshape sensor. The sensor by default looks at the +x direction.
        The field of view angles span (-FOV/2, 0) U (0, FOV/2) degrees
        """
        fov = params.get("fov", 90)
        min_range = params["min_range"]
        max_range = params["max_range"]
        self.name = name
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        # The size of the sensing region here is the area covered by the fan
        # This is a float, but rounding it up should equal to the number of discrete locations
        # in the field of view.
        self._sensing_region_size = int(math.ceil(self.fov / (2*math.pi) * math.pi * (max_range - min_range)**2))

    def uniform_sample_sensor_region(self, robot_pose):
        """Returns a location in the field of view
        uniformly at random. Expecting robot pose to
        have x, y, th, where th is in radians."""
        assert len(robot_pose) == 3,\
            "Robot pose must have x, y, th"
        # Sample a location (r,th) for the default robot pose
        th = random.uniform(0, self.fov) - self.fov/2
        r = random.uniform(self.min_range, self.max_range)
        x, y = pol2cart(r, th)
        # transform to robot pose
        x, y = np.matmul(R2d(robot_pose[2]), np.array([x,y])) # rotation
        x += robot_pose[0]  # translation dx
        y += robot_pose[1]  # translation dy
        point = (x, y)
        return point

    @property
    def sensor_region_size(self):
        return self._sensing_region_size

    def in_range(self, point, robot_pose):
        """
        Args:
            point (x, y): 2D point
            robot_pose (x, y, th): 2D robot pose
        """
        if robot_pose[:2] == point and self.min_range == 0:
            return True

        dist, bearing = self.shoot_beam(robot_pose, point)
        if self.min_range <= dist <= self.max_range:
            # because we defined bearing to be within 0 to 360, the fov
            # angles should also be defined within the same range.
            fov_ranges = (0, self.fov/2), (2*math.pi - self.fov/2, 2*math.pi)
            if in_range_inclusive(bearing, fov_ranges[0])\
               or in_range_inclusive(bearing, fov_ranges[1]):
                return True
            else:
                return False
        return False

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)
