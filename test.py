from math import pi

import numpy as np
from liegroups import SO3


r = pi / 6
p = pi / 7
y = pi / 4
rot = SO3.from_rpy(r, p, y)


ax1 = rot
array([[ 0.6370812 , -0.45897137,  0.61925183],
       [ 0.6370812 ,  0.7657735 , -0.08785495],

