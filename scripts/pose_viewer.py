"""Set thu cong 1 joint position va visualize robot VR-M3-1 bang launch_passive.

Cach dung:
    python scripts/pose_viewer.py
"""

import time
import mujoco
import mujoco.viewer

XML_PATH = "src/assets/robots/vr_m3_1/xmls/vr_m3_1.xml"

# Khop can set va goc (radian).
JOINT_NAME = "left_knee_pitch_joint"
JOINT_ANGLE = 1.00


def main() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Set thu cong 1 joint: ghi vao dung dia chi qpos cua khop.
    adr = model.jnt_qposadr[model.joint(JOINT_NAME).id]
    data.qpos[adr] = JOINT_ANGLE

    mujoco.mj_forward(model, data)  # cap nhat kinematics theo qpos vua set

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Chi forward (khong step) -> giu nguyen pose, khong nga.
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
