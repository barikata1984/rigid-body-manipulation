from omegaconf import OmegaConf

from trajectories.fourier import FourierTrajectoryConfig

# 1. Untyped
untyped_cfg = OmegaConf.load("configurations/trajectory_generation/fourier_6dof.yaml")
print(f"1. Untyped Config PRINT:\n{untyped_cfg}")
print(f"   -> to_object result: {type(OmegaConf.to_object(untyped_cfg))}")

# 2. Typed
typed_cfg = OmegaConf.merge(OmegaConf.structured(FourierTrajectoryConfig), untyped_cfg)
print(f"\n2. Typed Config PRINT:\n{typed_cfg}")
print(f"   -> to_object result: {type(OmegaConf.to_object(typed_cfg))}")

# 3. Direct Class Merge (what main.py does)
direct_class_cfg = OmegaConf.merge(FourierTrajectoryConfig, untyped_cfg)
print(f"\n3. Direct Class Merge PRINT:\n{direct_class_cfg}")
print(f"   -> to_object result: {type(OmegaConf.to_object(direct_class_cfg))}")
print(f"   -> has setup?: {hasattr(OmegaConf.to_object(direct_class_cfg), 'setup')}")
