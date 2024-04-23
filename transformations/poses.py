from liegroups import SE3
from mujoco._structs import MjData, MjModel

from utilities import get_element_id
from .transformations import compose


class Poses:
    def __init__(self,
                m: MjModel,
                d: MjData,
                ) -> None:
        self.m = m
        self.a_b = compose(m.body_pos, m.body_quat)
        self.b_bi = compose(m.body_ipos, m.body_iquat)
        self.x_b = compose(d.xpos, d.xmat)
        self.x_bi = compose(d.xipos, d.ximat)
        self.x_cam = compose(d.cam_xpos, d.cam_xmat)
        self.x_site = compose(d.site_xpos, d.site_xmat)


    def a_(self,
           name,
           ) -> SE3:

        return self.a_b[get_element_id(self.m, "body", name)]

    def b_principalof(self,
                      name,
                      ) -> SE3:

        return self.b_bi[get_element_id(self.m, "body", name)]

    def x_(self,
           name,
           type_,
           ) -> SE3:

        if "body" == type_:
            return self.x_b[get_element_id(self.m, type_, name)]
        elif "pricipal" == type_:
            return self.x_bi[get_element_id(self.m, "body", name)]
        elif "camera" == type_:
            return self.x_cam[get_element_id(self.m, type_, name)]
        elif "site" == type_:
            return self.x_site[get_element_id(self.m, type_, name)]
        else:
            raise ValueError(f"Pose retrievazl fo element type {type_} is not supported for now")
