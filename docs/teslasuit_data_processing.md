# Xử lý dữ liệu Teslasuit → puppet M3 (kinematic playback)

Tài liệu mô tả cách dữ liệu mocap Teslasuit được đọc, làm sạch và ánh xạ vào
robot VR-M3-1 trong MuJoCo. Pipeline được cài trong `scripts/teslasuit_puppet.py`.

- **Nguồn dữ liệu:** `puppet_data/data.zip` → giải nén ra `puppet_data/extracted/data.csv`
- **Model đích:** `src/assets/robots/vr_m3_1/xmls/vr_m3_1.xml` (full 27 DoF)
- **Chế độ:** kinematic playback (`mj_forward`, tắt gravity) — chỉ hiển thị lại
  chuyển động người, **không** mô phỏng động lực học, robot không ngã.

---

## 1. Cấu hình thời gian (repo & data)

| Thông số | Giá trị | Nguồn |
|---|---|---|
| Physics timestep | 0.005 s → **200 Hz** | `env_cfgs.py`: `MujocoCfg(timestep=0.005)` |
| Decimation | 4 | `env_cfgs.py`: `decimation=4` |
| Control/policy step | 0.02 s → **50 Hz** | `step_dt = timestep × decimation` |
| Teslasuit capture (nominal) | **50 Hz** (chọn được 28/50/100/200) | tài liệu Teslasuit |
| Data đo thực tế | dt ≈ 20.9 ms → **~47.8 Hz** | phân tích `data.csv` |

**Align:** capture nominal 50 Hz trùng control 50 Hz → **không cần resample** cho
playback. Chênh 47.8 Hz chỉ là jitter khi ghi log (std 0.27 ms, không gap). Chỉ
resample về lưới 20 ms đều khi bơm vào pipeline điều khiển/tracking 50 Hz.

---

## 2. Cấu trúc file `data.csv`

- **3977 frame × 284 cột**, ~83 giây.
- Cột `time` (giây) + các nhóm cột theo bone, mỗi nhóm gồm `position_0..2` và/hoặc
  `quaternion_0..3`.
- Có 3 loại cột:
  1. **Mocap thô** — `TS_Chest`, `TS_LeftFoot`, `TS_Hips`… (skeleton người) → **KHÔNG dùng**.
  2. **Trung gian** — `ts_spine2`, `ts_left_hip`, `*_angular_x_joint`… → **KHÔNG dùng**.
  3. **Đã retarget sang M3** — `ts_teleop_*` → **CHỈ DÙNG NHÓM NÀY**.

### Nhóm `ts_teleop_*` (đầu vào thực sự)

| Cột | Ý nghĩa | Map tới |
|---|---|---|
| `ts_teleop_<joint>_joint:joint_angular_position` | góc 27 khớp M3 (radian) | `qpos[7:34]` |
| `ts_teleop_pelvis:position_0..2` | vị trí base (Z-up, mét) | `qpos[0:3]` |
| `ts_teleop_pelvis:quaternion_0..3` | hướng base, **w-first** `[w,x,y,z]` | `qpos[3:7]` |

Tên khớp trong cột trùng **đúng** tên khớp M3 (vd `ts_teleop_left_hip_pitch_joint`
→ khớp `left_hip_pitch_joint`), nên ánh xạ theo **tên**, không đếm chỉ số.

---

## 3. Kiểm định chất lượng dữ liệu

| Hạng mục | Kết quả | Ghi chú |
|---|---|---|
| NaN / thiếu | **0** ở mọi cột | data đầy đủ |
| dt | mean 20.9 ms, std 0.27 ms | không gap, không dt ≤ 0 |
| Đơn vị góc | **radian** | dải trùng logic limit M3 |
| Base height (pelvis z) | ~1.005 m, gần cố định | ⇒ hệ **Z-up** |
| Quaternion norm | 1.0000 | đã chuẩn hóa sẵn |
| Thứ tự quaternion | **w-first** `[w,x,y,z]` | q0≈0.999 ở frame 0 (gần identity) |

Convention (Z-up, quaternion w-first, tên khớp M3) đã đúng chuẩn MuJoCo → dữ liệu
được xuất chủ đích để lái model M3.

---

## 4. Các bước xử lý (cleaning)

1. **Giải nén** `data.zip` → `data.csv`.
2. **Chọn cột:** chỉ giữ 27 cột `ts_teleop_*_joint:joint_angular_position` +
   `ts_teleop_pelvis` (pos 3 + quat 4). Bỏ ~250 cột mocap thô/trung gian.
3. **Ánh xạ theo tên:** với mỗi cột khớp, tách `joint_name`, tra
   `model.joint(joint_name).id` → `model.jnt_qposadr[jid]` để ghi đúng ô `qpos`.
4. **Clip biên khớp:** nếu `model.jnt_limited[jid]`, kẹp giá trị về
   `model.jnt_range[jid]`. Cần thiết vì một số khớp tay vượt biên M3:

   | Khớp | % frame vượt limit | Ghi chú |
   |---|---|---|
   | `left_elbow_pitch` | 43.1% | max ~1.66 > 1.571 |
   | `right_elbow_pitch` | 34.3% | max ~1.67 > 1.571 |
   | `right_shoulder_roll` | 29.4% | lố nhẹ qua mốc 0 |
   | `left_shoulder_roll` | 15.0% | lố nhẹ qua mốc 0 |
   | `right_shoulder_pitch` | 0.1% | không đáng kể |

   Chân / waist / hip / wrist: **sạch, 0% vượt**.
5. **Base:** ghi `qpos[0:3]` = pelvis position (`+ z_offset` tùy chọn),
   `qpos[3:7]` = quaternion đã normalize. Tùy chọn `z_offset ≈ -0.15` để hạ base
   (người ~1.0 m vs home M3 z≈0.854) cho chân sát sàn.
6. **Playback:** mỗi frame set `qpos` → `mj_forward` → `viewer.sync`, nhịp theo
   cột `time` thật (mốc tích lũy, tránh trôi). Tắt gravity.

**Không cần** ở chế độ playback: resample tần số, hiệu chỉnh T-pose, đổi hệ tọa độ
— vì data đã ở khung M3/MuJoCo.

---

## 5. Chạy

```bash
python scripts/teslasuit_puppet.py                    # full 27 DoF, real-time, clip on
python scripts/teslasuit_puppet.py --loop             # lặp lại
python scripts/teslasuit_puppet.py --z-offset -0.15   # chân sát sàn
python scripts/teslasuit_puppet.py --speed 0.3        # phát chậm để soi
python scripts/teslasuit_puppet.py --no-base          # giữ base cố định, chỉ xem khớp
python scripts/teslasuit_puppet.py --fixed-dt 0.02    # ép nhịp 50 Hz đều
```

Cờ khác: `--no-clip`, `--csv <path>`, `--xml <path>`.

Khi chạy, script in: số khớp map được (27/27), Hz capture (~47.8), số giá trị bị clip.

**Tài nguyên:** rất nhẹ — `mj_forward` chạy CPU (không phải mujoco-warp GPU),
GPU chỉ dùng để render 1 robot. Không tốn VRAM/compute như `train.py`.

---

## 6. Giới hạn của "naive plug" & hướng nâng cấp

- Playback **không có contact/thăng bằng** — chân có thể xuyên/lơ lửng sàn, chỉ là
  hiển thị động học. Muốn robot *thật sự* đứng và bám chuyển động dưới vật lý →
  cần **retarget → motion `.npz` → train tracking policy** (`MotionCommandCfg`
  trong mjlab), không thể cắm thẳng vào chế độ động lực học.
- Nếu cần đưa vào tracking 50 Hz: thêm bước **resample về lưới 20 ms đều** (slerp
  cho quaternion, nội suy tuyến tính cho góc khớp).
- Clip biên làm mất chi tiết ở khuỷu tay/vai khi người gập quá tầm robot — đây là
  giới hạn cơ khí của M3, không phải lỗi dữ liệu.
