"""Puppet M3 (full 27 DoF) bang du lieu mocap Teslasuit da retarget san.

Data (data.csv) chua nhom cot da retarget thang sang khung MuJoCo cua M3:
  - ts_teleop_<m3_joint>:joint_angular_position  -> goc 27 khop (radian)
  - ts_teleop_pelvis:position_0..2               -> vi tri base (Z-up, met)
  - ts_teleop_pelvis:quaternion_0..3             -> huong base [w, x, y, z]

Day la "naive plug" o che do KINEMATIC PLAYBACK:
  set qpos -> mj_forward (khong tich phan) -> viewer.sync.
Khong mo phong dong luc, tat gravity, nen robot khong nga; chi hien thi lai
chuyen dong nguoi.

Cach dung:
    python scripts/teslasuit_puppet.py
    python scripts/teslasuit_puppet.py --csv puppet_data/extracted/data.csv --speed 1.0 --loop
    python scripts/teslasuit_puppet.py --z-offset -0.15   # ha base cho chan sat san
    python scripts/teslasuit_puppet.py --video out/puppet.mp4 --no-viewer
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_XML = "src/assets/robots/vr_m3_1/xmls/vr_m3_1.xml"
DEFAULT_CSV = "puppet_data/extracted/data.csv"

TELEOP_PREFIX = "ts_teleop_"
JOINT_SUFFIX = ":joint_angular_position"


def load_csv(path):
    """Doc CSV -> (header: list[str], data: np.ndarray[float])."""
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return header, data


def build_joint_mapping(model, col):
    """Ghep cot teleop -> dia chi qpos cua tung khop M3.

    Tra ve list (qpos_adr, col_idx, lo, hi, joint_name).
    """
    mapping = []
    for name, idx in col.items():
        if not (name.startswith(TELEOP_PREFIX) and name.endswith(JOINT_SUFFIX)):
            continue
        joint_name = name[len(TELEOP_PREFIX):-len(JOINT_SUFFIX)]  # vd left_hip_pitch_joint
        try:
            jid = model.joint(joint_name).id
        except KeyError:
            print(f"[WARN] khop '{joint_name}' khong co trong model, bo qua")
            continue
        adr = model.jnt_qposadr[jid]
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
        else:
            lo, hi = -np.inf, np.inf
        mapping.append((adr, idx, lo, hi, joint_name))
    return mapping


def get_base_cols(col):
    """Lay chi so cot cho base (pelvis): (pos[3], quat[4]) hoac None."""
    try:
        pos = [col[f"ts_teleop_pelvis:position_{i}"] for i in range(3)]
        quat = [col[f"ts_teleop_pelvis:quaternion_{i}"] for i in range(4)]
        return pos, quat
    except KeyError:
        return None, None


def build_camera(model, args):
    """Chon camera cho video.

    Uu tien --video-camera (ten camera trong XML). Neu khong co, dung camera bam
    theo body --video-track: base chay theo mocap suot chuoi nen camera dung yen
    se de mat robot khoi khung hinh.
    """
    if args.video_camera is not None:
        return args.video_camera
    if not args.video_track:
        return -1
    try:
        bid = model.body(args.video_track).id
    except KeyError:
        print(f"[WARN] body '{args.video_track}' khong co trong model, dung camera tu do")
        return -1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = bid
    cam.distance = args.video_distance
    cam.azimuth = args.video_azimuth
    cam.elevation = args.video_elevation
    return cam


def render_video(model, data, args, apply_frame, n, t):
    """Render offscreen tung frame va ghi ra file video."""
    import imageio.v2 as imageio

    # Offscreen framebuffer trong XML thuong nho hon do phan giai yeu cau.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.video_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.video_height)

    if args.video_fps is not None:
        fps = args.video_fps
    else:
        if t is not None and len(t) > 1 and t[-1] > t[0]:
            capture_hz = (len(t) - 1) / (t[-1] - t[0])
        else:
            capture_hz = 1.0 / (args.fixed_dt or 0.02)
        # --speed nhanh hon => video co fps cao hon, giu nguyen so frame.
        fps = capture_hz * args.speed

    out = Path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] video      : {out}  ({args.video_width}x{args.video_height} @ {fps:.1f} fps)")

    camera = build_camera(model, args)
    with mujoco.Renderer(model, height=args.video_height, width=args.video_width) as renderer:
        with imageio.get_writer(out, fps=fps, macro_block_size=None) as writer:
            for i in range(n):
                apply_frame(i)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                if (i + 1) % 200 == 0 or i + 1 == n:
                    print(f"[INFO] rendered   : {i + 1}/{n} frames", end="\r", flush=True)
    print()
    print(f"[INFO] saved      : {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--speed", type=float, default=1.0, help="he so toc do phat (1.0 = real-time theo cot time)")
    p.add_argument("--z-offset", type=float, default=0.0, help="cong them vao base z (met), vd -0.15 de chan sat san")
    p.add_argument("--no-clip", action="store_true", help="khong clip goc khop ve joint limit")
    p.add_argument("--no-base", action="store_true", help="giu base co dinh, chi nhai goc khop")
    p.add_argument("--fixed-dt", type=float, default=None, help="ep chu ky phat (giay), thay cho cot time (vd 0.02)")
    p.add_argument("--loop", action="store_true", help="lap lai khi het chuoi")
    p.add_argument("--video", default=None, help="xuat video ra duong dan nay (vd out/puppet.mp4)")
    p.add_argument("--video-fps", type=float, default=None, help="fps video (mac dinh: suy tu cot time)")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument("--video-camera", default=None, help="ten camera trong model (mac dinh: camera bam theo pelvis)")
    p.add_argument("--video-track", default="pelvis", help="body de camera bam theo (rong = camera tu do dung yen)")
    p.add_argument("--video-distance", type=float, default=2.5, help="khoang cach camera bam (met)")
    p.add_argument("--video-azimuth", type=float, default=135.0)
    p.add_argument("--video-elevation", type=float, default=-15.0)
    p.add_argument("--no-viewer", action="store_true", help="khong mo cua so viewer (chi xuat video)")
    args = p.parse_args()

    if args.no_viewer and not args.video:
        p.error("--no-viewer can di kem --video, neu khong se khong hien thi gi")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Kinematic playback: tat gravity cho chac (du mj_forward von khong tich phan).
    model.opt.gravity[:] = 0.0

    header, csv = load_csv(args.csv)
    col = {name: i for i, name in enumerate(header)}

    mapping = build_joint_mapping(model, col)
    pos_cols, quat_cols = get_base_cols(col)
    has_base = (pos_cols is not None) and (not args.no_base)

    # Timeline.
    t = csv[:, col["time"]] if "time" in col else None
    n = csv.shape[0]

    print(f"[INFO] model      : {args.xml}")
    print(f"[INFO] csv        : {args.csv}  ({n} frames)")
    print(f"[INFO] mapped     : {len(mapping)}/27 khop teleop -> qpos")
    print(f"[INFO] base drive : {has_base}  (z_offset={args.z_offset})")
    print(f"[INFO] clip limit : {not args.no_clip}   speed={args.speed}   loop={args.loop}")
    if t is not None and len(t) > 1:
        print(f"[INFO] capture    : ~{(len(t)-1)/(t[-1]-t[0]):.1f} Hz, duration {t[-1]-t[0]:.1f}s")

    # Dem so lan clip de bao chat luong.
    if not args.no_clip:
        n_clip = 0
        for adr, idx, lo, hi, _ in mapping:
            v = csv[:, idx]
            n_clip += int(np.sum((v < lo) | (v > hi)))
        print(f"[INFO] clipped    : {n_clip} gia tri khop bi dua ve trong limit")

    def apply_frame(i):
        # Goc khop.
        for adr, idx, lo, hi, _ in mapping:
            val = csv[i, idx]
            if not args.no_clip:
                val = min(max(val, lo), hi)
            data.qpos[adr] = val
        # Base (khop free: qpos[0:3] pos, qpos[3:7] quat [w,x,y,z]).
        if has_base:
            data.qpos[0] = csv[i, pos_cols[0]]
            data.qpos[1] = csv[i, pos_cols[1]]
            data.qpos[2] = csv[i, pos_cols[2]] + args.z_offset
            q = np.array([csv[i, c] for c in quat_cols], dtype=float)
            nrm = np.linalg.norm(q)
            if nrm > 1e-8:
                q /= nrm
            data.qpos[3:7] = q
        mujoco.mj_forward(model, data)

    apply_frame(0)

    if args.video:
        render_video(model, data, args, apply_frame, n, t)

    if not args.no_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                wall0 = time.perf_counter()
                for i in range(n):
                    if not viewer.is_running():
                        break
                    apply_frame(i)
                    viewer.sync()

                    # Nhip phat: mo thoi gian tich luy tu dau chuoi (tranh troi tich luy).
                    # Dung cot time thuc neu co, nguoc lai dung fixed-dt (mac dinh 20ms).
                    if i + 1 < n:
                        if t is not None and args.fixed_dt is None:
                            elapsed = t[i + 1] - t[0]
                        else:
                            elapsed = (i + 1) * (args.fixed_dt or 0.02)
                        target = wall0 + elapsed / max(args.speed, 1e-6)
                        sleep = target - time.perf_counter()
                        if sleep > 0:
                            time.sleep(sleep)
                if not args.loop:
                    break

    print("[INFO] done.")


if __name__ == "__main__":
    main()
