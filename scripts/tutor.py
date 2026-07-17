import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("src\\assets\\robots\\vr_m3_1\\xmls\\vr_m3_1.xml")
data = mujoco.MjData(model)

print("nq (generalized pos):", model.nq)
print("nv (generalized vel):", model.nv)
print("nu (actuators):", model.nu)
print("Total mass (kg):", model.body_mass.sum())

print("Joint names:", [model.joint(i).name for i in range(model.njnt)])
print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("Base position:", data.qpos[:3])
print("Base quaternion:", data.qpos[3:7])
print("Joint positions:", data.qpos[7:])