# This is a toy domain for 2D COS-POMDP
from cospomdp_apps.basic.common import solve

WORLD =\
"""
### map
R.G..
.x.Tx
.x..x

### robotconfig
th: 0

### corr
T around G: d=2

### detectors
T: fan-nofp | fov=45, min_range=0, max_range=2 | (0.6, 0.1)
G: fan-nofp | fov=45, min_range=0, max_range=3 | (0.8, 0.1)

### goal
nav: T

### colors
T: [0, 22, 120]
G: [0, 210, 20]

### END
"""

if __name__ == "__main__":
    solve(WORLD, nsteps=50,
          solver="pomdp_py.POUCT",
          solver_args=dict(max_depth=15,
                           planning_time=1.,
                           discount_factor=0.95,
                           exploration_const=100))
