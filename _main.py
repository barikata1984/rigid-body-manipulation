import copy
from pathlib import Path

import mujoco as mj
from dm_control import mjcf
from dm_control.mjcf import RootElement

from hydra_zen import builds, load_from_yaml, make_config, store, zen


# Configure manipulaotrs and target objects =======================================
manipulator_store = store(group="manipulator")
manipulator_store(mjcf.from_path, path="xml_models/manipulator.xml", name="relative")

target_object_store = store(group="target_object")
target_object_store(mjcf.from_path, path="xml_models/uniform123_128.xml", name="uniform123_128")


# Make a default hydra configuraiton ==============================================
store(
    make_config(
        hydra_defaults=["_self_", {"manipulator": "relative"},
                                  {"target_object": "uniform123_128"},
                        ],
        manipulator=None,
        target_object=None,
    ),
    name="config",
)


def simulate(manipulator: RootElement,  #: Manipulator,  # actually, it can be MjModel
             target_object: RootElement,  # actually, it can be MjModel
#             trajectory: Trajectory,  # not sure if this should be a class, just for now
#             planner: Planner,  # should be implemented by myself
#             visualizer: Visualizer,  #
             ):

    manipulation = copy.deepcopy(manipulator)
    attachement_site = manipulation.find('site', 'attachment')
    attachement_site.attach(target_object)

    m = mj.MjModel.from_xml_string(manipulation.to_xml_string())

    print(m)

    return


if __name__ == "__main__":
    from hydra.conf import HydraConf, JobConf
    # Activating the following line change the current working directory to the
    # "otput" directory
    #store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")

    # Add all of the configs, that we put in hydra-zen's (local) config store,
    # to Hydra's (global) config store.
    store.add_to_hydra_store(overwrite_ok=True)

    # Use
    # - `zen()` to convert our Hydra-agnostic task function into one that is
    # compatible with Hydra.
    # - `.hydra_main(...)` to generate the Hydra-compatible CLI for our program.
    zen(simulate).hydra_main(config_path=None,
                             config_name="config",
                             version_base="1.3",
                             )


