# Báo cáo phân tích robot VR-M3-1

Nguồn: `src/assets/robots/vr_m3_1/xmls/vr_m3_1.xml`, `src/assets/robots/vr_m3_1/vr_m3_1_constants.py`, chạy `scripts/tutor.py`.

---

## 1. Số khớp được điều khiển & giới hạn position/velocity

Robot có **27 khớp được điều khiển** (actuated joints). Ngoài ra còn `base_joint` (khớp `free` của thân — 6 DoF, không điều khiển) và 2 khớp đầu (`head_yaw`, `head_pitch`) **đã bị comment** trong XML nên không tồn tại.

> Lưu ý: `tutor.py` in ra `nu (actuators): 0` vì file XML thô **không có block `<actuator>`**. Các actuator được mjlab thêm vào lúc runtime qua `BuiltinPositionActuatorCfg` trong `vr_m3_1_constants.py` (14 nhóm regex → 27 khớp).

- **Giới hạn position**: lấy từ thuộc tính `range` trong XML.
- **Giới hạn velocity**: XML không có; lấy từ `MOTOR_SPECS[...]["max_vel"]` trong `constants.py` (ánh xạ qua `VR_M3_1_ACTUATOR_NAMES`).

| Khớp (×2 trái/phải trừ waist) | Position range (rad) | Velocity max (rad/s) | Motor | Torque (N·m) |
|---|---|---|---|---|
| hip_pitch | -2.617 … 2.617 | 14.653 | LEG_MOTOR_1 | ±360 |
| hip_roll (L) | -0.785 … 2.094 | 14.653 | LEG_MOTOR_1 | ±360 |
| hip_roll (R) | -2.094 … 0.785 | 14.653 | LEG_MOTOR_1 | ±360 |
| hip_yaw | -2.617 … 2.617 | 31.4 | LEG_MOTOR_2 | ±130 |
| knee_pitch | -0.174 … 2.356 | 14.653 | LEG_MOTOR_1 | ±360 |
| ankle_pitch | -0.873 … 0.523 | 16.747 | LEG_MOTOR_3 | ±120 |
| ankle_roll | -0.262 … 0.262 | 16.747 | LEG_MOTOR_3 | ±120 |
| waist_yaw | -3.14 … 3.14 | 4.18 | WAIST_MOTOR | ±102 |
| shoulder_pitch | -3.14 … 1.57 | 4.29 | ARM_MOTOR_1 | ±66 |
| shoulder_roll (L) | 0 … 2.356 | 4.29 | ARM_MOTOR_1 | ±66 |
| shoulder_roll (R) | -2.356 … 0 | 4.29 | ARM_MOTOR_1 | ±66 |
| shoulder_yaw | -2.79 … 2.79 | 5.13 | ARM_MOTOR_2 | ±34 |
| elbow_pitch | -0.524 … 1.571 | 5.13 | ARM_MOTOR_2 | ±34 |
| wrist_yaw | -2.79 … 2.79 | 6.17 | ARM_MOTOR_3 | ±11 |
| wrist_roll | -1.571 … 1.571 | 6.17 | ARM_MOTOR_3 | ±11 |
| wrist_pitch | -1.571 … 1.571 | 6.17 | ARM_MOTOR_3 | ±11 |

Phân bổ 27 khớp: chân 6×2 = 12, thân (waist) 1, tay 7×2 = 14.

---

## 2. IMU gắn ở đâu? Đo cái gì?

- **Vị trí**: `<site name="imu">` gắn trên body **`pelvis`** (xương chậu), tại `pos="0.00614 0 -0.02485"` (mét, trong hệ tọa độ pelvis).
- **Đo gì** (khai báo trong block `<sensor>`, gắn vào site `imu`):
  - `imu_ang_vel` — **gyro**: vận tốc góc (angular velocity).
  - `imu_lin_vel` — **velocimeter**: vận tốc dài (linear velocity).
  - `imu_lin_acc` — **accelerometer**: gia tốc dài (linear acceleration).
- Ngoài ra `root_angmom` — **subtreeangmom** trên body `pelvis`: mô-men động lượng của toàn bộ cây con (không phải sensor của IMU nhưng gắn ở gốc).

---

## 3. Khối lượng tổng của robot

`model.body_mass.sum()` = **57.335598 kg** (≈ 57.34 kg).

---

## 4. Giải thích từng dòng output của `scripts/tutor.py`

```
nq (generalized pos): 34
nv (generalized vel): 33
nu (actuators): 0
Total mass (kg): 57.335598
Joint names: ['base_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint']
Actuator names: []
Base position: [0. 0. 0.]
Base quaternion: [1. 0. 0. 0.]
Joint positions: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0.]
```

- **`nq = 34`**: số tọa độ vị trí suy rộng. = 7 (khớp `free` của thân: 3 vị trí + 4 quaternion) + 27 khớp hinge = 34.
- **`nv = 33`**: số vận tốc suy rộng. = 6 (thân free: 3 tịnh tiến + 3 quay) + 27 khớp hinge = 33. (Quaternion 4 số nhưng chỉ 3 DoF quay → nq lớn hơn nv đúng 1.)
- **`nu = 0`**: số actuator. Bằng 0 vì XML thô không có `<actuator>`; actuator do mjlab thêm lúc runtime (xem mục 1).
- **`Total mass = 57.335598`**: tổng khối lượng tất cả body (`model.body_mass.sum()`).
- **`Joint names`**: danh sách 28 khớp — `base_joint` (free) + 27 khớp hinge có điều khiển.
- **`Actuator names: []`**: rỗng, cùng lý do `nu = 0`.
- **`Base position: [0 0 0]`**: vị trí gốc thân sau `mj_resetData` (qpos[:3]) — về mặc định gốc tọa độ, chưa áp keyframe HOME (z=0.854 chỉ có trong config mjlab).
- **`Base quaternion: [1 0 0 0]`**: hướng thân (qpos[3:7]) — quaternion đơn vị, không xoay.
- **`Joint positions: [0 ... 0]`**: 27 góc khớp (qpos[7:]) — đều bằng 0 ở trạng thái reset mặc định.
