# Projections

def inverse_perspective(pixel_loc, camera_intrinsics, camera_extrinsics):
    """
    Maps a pixel to 3D world
    Args:
        pixel_loc (tuple): x, y
    """
    raise NotImplementedError

def perspective(loc3d, camera_intrinsics, camera_extrinsics):
    """
    Maps a 3D world point to a pixel location

    Args:
        loc3d (tuple): x, y, z
    """
    raise NotImplementedError
