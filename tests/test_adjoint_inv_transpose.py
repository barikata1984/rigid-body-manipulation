from math import pi

import numpy as np
from numpy import linalg as la
from liegroups.numpy import SO3, SE3


rot = SO3.from_rpy(10 / 180 * pi,
                   20 / 180 * pi,
                   40 / 180 * pi,
                   )

trans = np.array([1, 2, 3])
pose = SE3(rot, trans)


print(pose)

Ad = pose.adjoint()
Ad_T = Ad.T

Ad_inv = pose.inv().adjoint()
pinv_Ad = la.pinv(pose.adjoint())


print(f"{np.allclose(Ad_T, Ad_inv)=}")
print(f"{np.allclose(Ad_inv, pinv_Ad)=}")
