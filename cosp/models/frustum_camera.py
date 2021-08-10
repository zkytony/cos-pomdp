import math
import numpy as np
import moos3d.util as util
import random
from scipy.spatial.transform import Rotation as R

class SensorModel:
    def __init__(self):
        pass

class CameraModel(SensorModel):
    """Each observation is sets of voxels, grouped by objects (if known correspondence)"""
    def __init__(self):
        pass
    @abstractmethod
    def within_range(self, config, entity, **kwargs):
        """
        Returns True, if this camera model, when applied the given `config`,
        will result in entity within the field of view.
        """
        pass

class FrustumCamera(CameraModel):

    @property
    def near(self):
        return self._params[-2]

    @property
    def far(self):
        return self._params[-1]

    @property
    def fov(self):
        """returns fov in degrees"""
        return self._params[0] / math.pi * 180

    @property
    def aspect_ratio(self):
        return self._params[1]

    @property
    def volume(self):
        return self._volume

    def print_info(self):
        print("         FOV: " + str(self.fov))
        print("aspect_ratio: " + str(self.aspect_ratio))
        print("        near: " + str(self.near))
        print("         far: " + str(self.far))
        print(" volume size: " + str(len(self.volume)))

    def __init__(self, fov=90, aspect_ratio=1, near=1, far=5):
        """
        fov: angle (degree), how wide the viewing angle is.
        near: near-plane's distance to the camera
        far: far-plane's distance to the camera
        """
        # Initially, the camera is always at (0,0,0), looking at direction (0,0,-1)
        # This can be changed by calling `transform_camera()`
        #
        # 6 planes:
        #     3
        #  0 2 4 5
        #     1

        # sizes of near and far planes
        fov = fov*math.pi / 180
        h1 = near * math.tan(fov/2) * 2
        w1 = abs(h1 * aspect_ratio)
        h2 = far * math.tan(fov/2) * 2
        w2 = abs(h2 * aspect_ratio)
        self._dim = (w1, h1, w2, h2)
        self._params = (fov, aspect_ratio, near, far)

        ref1 = np.array([w1/2, h1/2, -near, 1])
        ref2 = np.array([-w2/2, -h2/2, -far, 1])

        p1A = np.array([w1/2, h1/2, -near])
        p1B = np.array([-w1/2, h1/2, -near])
        p1C = np.array([w1/2, -h1/2, -near])
        n1 = np.cross(util.vec(p1A, p1B),
                      util.vec(p1A, p1C))

        p2A = p1A
        p2B = p1C
        p2C = np.array([w2/2, h2/2, -far])
        n2 = np.cross(util.vec(p2A, p2B),util.vec(p2A, p2C))

        p3A = p1A
        p3B = p2C
        p3C = p1B
        n3 = np.cross(util.vec(p3A, p3B),
                      util.vec(p3A, p3C))

        p4A = np.array([-w2/2, -h2/2, -far])
        p4B = np.array([-w1/2, -h1/2, -near])
        p4C = np.array([-w2/2, h2/2, -far])
        n4 = np.cross(util.vec(p4A, p4B),
                      util.vec(p4A, p4C))

        p5A = p4B
        p5B = p4A
        p5C = p2B
        n5 = np.cross(util.vec(p5A, p5B),
                      util.vec(p5A, p5C))

        p6A = p4A
        p6B = p4C
        p6C = p2C
        n6 = np.cross(util.vec(p6A, p6B),
                      util.vec(p6A, p6C))

        p = np.array([n1,n2,n3,n4,n5,n6])
        for i in range(6):  # normalize
            p[i] = p[i] / np.linalg.norm(p[i])
        p = np.array([p[i].tolist() + [0] for i in range(6)])
        r = np.array([ref1, ref1, ref1, ref2, ref2, ref2])
        assert self.within_range((p, r), [0,0,-far-(-far+near)/2, 1])
        self._p = p
        self._r = r

        # compute the volume inside the frustum
        volume = []
        count = 0
        for z in range(-int(round(far)), -int(round(near))):
            for y in range(-int(round(h2/2))-1, int(round(h2/2))+1):
                for x in range(-int(round(w2/2))-1, int(round(w2/2))+1):
                    if self.within_range((self._p, self._r), (x,y,z,1)):
                        volume.append([x,y,z,1])
        self._volume = np.array(volume, dtype=int)

    def within_range(self, config, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion"""
        p, r = config
        for i in range(6):
            if np.dot(util.vec(r[i], point), p[i]) >= 0:
                # print("Point outside plane %i" % i)
                # print("    Plane normal: %s" % str(p[i]))
                # print("    Plane refs: %s" % str(r[i]))
                # print("       Measure: %.3f" % np.dot(util.vec(r[i], point), p[i]))
                return False
        return True


    def perspectiveTransform(self, x, y, z, camera_pose):
        # @params:
        # - x,y,z: points in world space
        # - camera_pose: [eye_x, eye_y, eye_z, theta_x, theta_y, theta_z]

        point_in_world = [x,y,z,1.0]
        eye = camera_pose[:3]
        rot = camera_pose[3:]

        #default up and look vector when camera pose is (0,0,0)
        up = np.array([0.0, 1.0, 0.0])
        look = np.array([0.0, 0.0, -1.0])

        #transform up, look vector according to current camera pose
        r = R.from_quat([rot[0],rot[1],rot[2],rot[3]])
        curr_up = r.apply(up)
        curr_look = r.apply(look)
        curr_up += eye
        curr_look += eye

        #derive camera space axis u,v,w -> lookat Matrix
        w = - (curr_look - eye) / np.linalg.norm(curr_look - eye)
        v = curr_up - np.dot(curr_up, w) * w
        v = v / np.linalg.norm(v)
        u = np.cross(v,w)
        lookat = np.array([u, v, w])

        #Transform point in World Space to perspective Camera Space
        mat = np.eye(4)
        mat[0, 3] = -eye[0]
        mat[1, 3] = -eye[1]
        mat[2, 3] = -eye[2]
        point_in_camera = np.matmul(mat, point_in_world)

        axis_mat = np.eye(4)
        axis_mat[:3, :3] = lookat
        point_in_camera = np.matmul(axis_mat, point_in_camera)

        #Transform point in perspective Camera Space to normalized perspective Camera Space
        p_norm =  1.0 / ( self._params[3]  * np.tan(( self._params[0] * (np.pi/180.0) )/2) )
        norm_mat = np.eye(4, dtype = np.float32)
        norm_mat[0, 0] = p_norm
        norm_mat[1, 1] = p_norm
        norm_mat[2, 2] = 1.0 / self._params[-1]
        point_in_norm = np.matmul(norm_mat, point_in_camera)

        #Transform point in normalized perspective Camera Space to parallel camera viewing space
        c = - self._params[2] / self._params[3]
        unhinge_mat = np.eye(4, dtype=np.float32)
        unhinge_mat[2,2] = -1.0 / (1+c)
        unhinge_mat[2,3] = c / (1+c)
        unhinge_mat[3,2] = -1.0
        unhinge_mat[3,3] = 0.0
        point_in_parallel = np.matmul(unhinge_mat, point_in_norm)

        #De-homogenize
        point_in_parallel = point_in_parallel/ point_in_parallel[-1]

        return point_in_parallel

    def get_config(self):
        return self._p, self._r

    def transform_camera(self, pose, permanent=False):#x, y, z, thx, thy, thz, permanent=False):
        """Transformation relative to current pose; Affects where the sensor's field of view.
        thx, thy, thz are in degrees. Returns the configuration after the transform is applied.
        In other words, this is saying `set up the camera at the given pose`."""
        if len(pose) == 7:
            x, y, z, qx, qy, qz, qw = pose
            R = util.R_quat(qx, qy, qz, qw, affine=True)
        elif len(pose) == 6:
            x, y, z, thx, thy, thz = pose
            R = util.R_euler(thx, thy, thz, affine=True)
        r_moved = np.transpose(np.matmul(util.T(x, y, z),
                                         np.matmul(R, np.transpose(self._r))))
        p_moved =  np.transpose(np.matmul(R, np.transpose(self._p)))
        if permanent:
            self._p = p_moved
            self._r = r_moved
            self._volume = np.transpose(np.matmul(util.T(x, y, z),
                                                  np.matmul(R, np.transpose(self._volume))))
        return p_moved, r_moved


    # We need the notion of free space. The simplest thing to do
    # is for the sensor to directly inform the robot what the free
    # spaces are.
    def get_volume(self, robot_pose, volume=None):
        """Return the volume inside the frustum as a list of 3D coordinates."""
        if volume is None:
            volume = self._volume
        if len(robot_pose) == 7:
            x, y, z, qx, qy, qz, qw = robot_pose
            R = util.R_quat(qx, qy, qz, qw, affine=True)
        elif len(robot_pose) == 6:
            x, y, z, thx, thy, thz = robot_pose
            R = util.R_euler(thx, thy, thz, affine=True)
        volume_moved = np.transpose(np.matmul(util.T(x, y, z),
                                              np.matmul(R, np.transpose(volume))))
        # Get x,y,z only
        volume_moved = volume_moved[:,:3]
        return np.round(volume_moved).astype(int)

    def field_of_view_size(self):
        return len(self._volume)

    def get_direction(self, p=None):
        if p is None:
            return -self._p[0][:3]
        else:
            return -p[0][:3]

    @staticmethod
    def sensor_functioning(alpha=1000., beta=0., log=False):
        """Utility used when sampling observation, to determine if the sensor works properly.
        (i.e. observed = True if the sensor works properly)"""
        if log:
            # e^a / (e^a + e^b) = 1 / (e^{b-a} + 1)
            observed = random.uniform(0,1) < 1 / (math.exp(beta - alpha) + 1)
        else:
            observed = random.uniform(0,1) < alpha / (alpha + beta)
        return observed
