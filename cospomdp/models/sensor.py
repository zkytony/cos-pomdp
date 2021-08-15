class SensorModel:
    """Sensor. Not tied to any particular object"""
    def in_range(self, robot_pose, object_pose):
        raise NotImplementedError

    def sensor_region(self, robot_pose):
        raise NotImplementedError

    @property
    def sensor_region_size(self):
        raise NotImplementedError

    def uniform_sample_sensor_region(self, robot_pose):
        """Returns a location in the field of view
        uniformly at random"""
        raise NotImplementedError

    def within_range(self, config, entity, **kwargs):
        """
        Returns True, if this camera model, when applied the given `config`,
        will result in entity within the field of view.
        """
        pass
