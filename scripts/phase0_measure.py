#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Giai đoạn 0 — ĐO, chưa tối ưu.

Tương đương lệnh:

    python scripts/train.py VR-M3-1-Stand --env.scene.num-envs=4096 \
        --agent.max-iterations=2000 --gpu-ids '[0, 1]'

LƯU Ý ĐA GPU: mỗi rank dựng ĐỦ ``num_envs`` riêng, nên 2 GPU × 4096 = 8192 env
tổng, và ``Perf/total_fps`` đã gộp sẵn mọi GPU
(``collection_size = num_steps_per_env * num_envs * world_size``,
rsl_rl/utils/logger.py:152). Script nhân đúng thừa số này khi quy đổi ngân sách
phút sang số iteration.

Chỉ khác hai điểm, cả hai đều là bổ sung chứ không đổi hành vi train:
  1. IN TOÀN BỘ thông số của phiên chạy ra màn hình + logs/phase0/ trước khi chạy.
  2. Sau khi chạy xong (hoặc khi bị Ctrl-C), đọc log và báo cáo hai con số
     quyết định:

     • Perf/total_fps          — thông lượng. Run cũ trong logs/ chạy num_envs=4
                                 và chỉ đạt 34 FPS, nên không học được gì.
     • Train/mean_episode_length — TÍN HIỆU SỐNG-CÒN. Phải ĐI LÊN.
                                 Run cũ tụt 57 → 10 rồi phẳng lì 4700 iter.

Mọi thứ khác giữ nguyên mặc định của repo: logger W&B, save_interval=1000
(nên vẫn có checkpoint ở iter 1000/2000), experiment_name=vr_m3_1_velocity,
log vẫn nằm ở logs/rsl_rl/<experiment>/<datetime>/ đúng format cũ.

Dùng:
    python scripts/phase0_measure.py --dry-run       # chỉ in thông số rồi thoát
    python scripts/phase0_measure.py                 # in thông số rồi train
    python scripts/phase0_measure.py --analyze latest # đọc lại run đã/đang chạy
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "logs" / "rsl_rl"
OUT_DIR = REPO / "logs" / "phase0"
BAR = "=" * 78


# --------------------------------------------------------------------------- #
# In thông số
# --------------------------------------------------------------------------- #
def _fmt(v, width=58):
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= width else s[: width - 3] + "..."


def print_run_params(task_id: str, args, cfg, agent) -> str:
    """In TOÀN BỘ thông số của phiên chạy này. Trả về text để lưu ra file."""
    lines: list[str] = []
    w = lines.append

    def sec(title):
        w("")
        w(f"--- {title} " + "-" * max(0, 74 - len(title)))

    def kv(k, v, note=""):
        w(f"  {k:<34} {_fmt(v):<26} {note}")

    w(BAR)
    w(f"  GIAI ĐOẠN 0 — ĐO  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    w(BAR)

    n_gpu = count_gpus(args.gpu_ids)
    sec("Lệnh tương đương")
    w(f"  python scripts/train.py {task_id} \\")
    w(f"      --env.scene.num-envs={args.num_envs} \\")
    w(f"      --agent.max-iterations={args.iters}" + (" \\" if args.gpu_ids else ""))
    if args.gpu_ids:
        w(f"      --gpu-ids '{args.gpu_ids}'")

    # --- Phần cứng ---
    sec("Phần cứng")
    try:
        import torch

        kv("torch", torch.__version__)
        kv("cuda available", torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            kv("GPU", torch.cuda.get_device_name(0))
            kv("VRAM tổng", f"{props.total_memory / 1024**3:.1f} GB")
            kv("compute capability", f"{props.major}.{props.minor}")
    except Exception as e:  # noqa: BLE001
        kv("torch", f"<không import được: {e}>")

    # --- Ngân sách của phiên chạy ---
    sec("Ngân sách phiên chạy")
    timestep = cfg.sim.mujoco.timestep
    dec = cfg.decimation
    step_dt = timestep * dec
    max_steps = int(cfg.episode_length_s / step_dt)
    total_steps = args.iters * agent.num_steps_per_env * args.num_envs * n_gpu
    kv("gpu_ids", args.gpu_ids or "(mặc định [0])", f"= {n_gpu} GPU")
    kv("num_envs MỖI GPU", args.num_envs, "<- GHI ĐÈ (config gốc: %s)" % cfg.scene.num_envs)
    if n_gpu > 1:
        kv("num_envs TỔNG", f"{args.num_envs * n_gpu:,}", "mỗi rank dựng đủ num_envs riêng")
    kv("max_iterations", args.iters, "<- GHI ĐÈ (config gốc: %s)" % agent.max_iterations)
    kv("num_steps_per_env", agent.num_steps_per_env)
    kv("mẫu / lần update PPO", f"{args.num_envs * agent.num_steps_per_env * n_gpu:,}")
    kv("TỔNG env-step", f"{total_steps:,}", f"= {total_steps / 1e6:.0f}M")
    w("")
    w("  Đối chiếu run cũ (logs/rsl_rl/vr_m3_1_12dof_velocity/2026-07-15_02-11-33):")
    w("      num_envs=4 · 34 FPS · 5254 iter · tổng 672,512 env-step")
    w(f"      → phiên này lớn hơn {total_steps / 672512:,.0f}× về dữ liệu")

    # --- Timing ---
    sec("Đồng hồ hệ thống (dẫn xuất)")
    kv("sim.mujoco.timestep", timestep, f"= {1 / timestep:.0f} Hz vật lý")
    kv("decimation", dec)
    kv("step_dt", f"{step_dt:.4f} s", f"= {1 / step_dt:.0f} Hz policy")
    kv("episode_length_s", cfg.episode_length_s, f"= {max_steps} bước/episode")

    # --- Scene / sim ---
    sec("Scene & Sim")
    kv("scene.extent", cfg.scene.extent)
    terrain = cfg.scene.terrain
    kv("terrain.terrain_type", getattr(terrain, "terrain_type", None))
    kv("terrain.terrain_generator", type(getattr(terrain, "terrain_generator", None)).__name__)
    kv("sim.nconmax", cfg.sim.nconmax)
    kv("sim.njmax", cfg.sim.njmax)
    kv("sim.contact_sensor_maxmatch", cfg.sim.contact_sensor_maxmatch)
    kv("sim.mujoco.iterations", cfg.sim.mujoco.iterations)
    kv("sim.mujoco.ls_iterations", cfg.sim.mujoco.ls_iterations)
    kv("sim.mujoco.ccd_iterations", cfg.sim.mujoco.ccd_iterations)

    # --- Action ---
    sec("Action")
    for name, term in cfg.actions.items():
        kv(f"actions[{name}]", type(term).__name__)
        kv("  .actuator_names", getattr(term, "actuator_names", None))
        kv("  .use_default_offset", getattr(term, "use_default_offset", None))
        scale = getattr(term, "scale", None)
        if isinstance(scale, dict):
            kv("  .scale", f"<dict {len(scale)} khớp>")
            for jn, sv in scale.items():
                w(f"      {jn:<40} {sv:.4f}")
        else:
            kv("  .scale", scale)

    # --- Observation ---
    sec("Observation")
    for gname, group in cfg.observations.items():
        terms = list(group.terms.keys())
        kv(f"observations[{gname}]", f"{len(terms)} term")
        for t in terms:
            tc = group.terms[t]
            w(
                f"      {t:<24} scale={_fmt(getattr(tc, 'scale', None), 10):<12}"
                f" noise={'có' if getattr(tc, 'noise', None) else '-':<4}"
                f" hist={getattr(tc, 'history_length', 0)}"
            )
        kv("  .enable_corruption", group.enable_corruption)

    # --- Command ---
    sec("Command")
    for name, term in cfg.commands.items():
        kv(f"commands[{name}]", type(term).__name__)
        for f in ("resampling_time_range", "rel_standing_envs", "heading_command"):
            if hasattr(term, f):
                kv(f"  .{f}", getattr(term, f))
        rng = getattr(term, "ranges", None)
        if rng is not None:
            for f in dataclasses.fields(rng):
                kv(f"  .ranges.{f.name}", getattr(rng, f.name))

    # --- Reward ---
    sec(f"Reward ({len(cfg.rewards)} term) — mọi weight còn được nhân step_dt")
    for name, term in sorted(cfg.rewards.items(), key=lambda it: -abs(it[1].weight)):
        fn = getattr(term.func, "__name__", type(term.func).__name__)
        w(f"  {name:<26} weight={term.weight:>12.4g}   func={fn}")

    # --- Termination ---
    sec("Termination")
    for name, term in cfg.terminations.items():
        fn = getattr(term.func, "__name__", type(term.func).__name__)
        w(f"  {name:<24} time_out={str(term.time_out):<6} {fn} {_fmt(term.params, 28)}")

    # --- Curriculum ---
    sec("Curriculum")
    if not cfg.curriculum:
        w("  (rỗng)")
    for name, term in cfg.curriculum.items():
        fn = getattr(term.func, "__name__", type(term.func).__name__)
        kv(f"curriculum[{name}]", fn)
        stages = term.params.get("velocity_stages")
        if stages:
            for st in stages:
                it = st["step"] / max(agent.num_steps_per_env, 1)
                reach = "" if it <= args.iters else "  <-- NGOÀI ngân sách phiên này"
                w(
                    f"      step={st['step']:>9,} (~iter {it:>6.0f})  "
                    f"x={st.get('lin_vel_x')} yaw={st.get('ang_vel_z')}{reach}"
                )

    # --- Domain randomization ---
    sec("Event / Domain randomization")
    by_mode: dict[str, list[str]] = {}
    for name, term in cfg.events.items():
        by_mode.setdefault(term.mode, []).append(name)
    for mode, names in by_mode.items():
        kv(f"mode='{mode}'", f"{len(names)} term")
        for n in names:
            p = cfg.events[n].params
            op = p.get("operation", "")
            rngs = {k: v for k, v in p.items() if k.endswith(("range", "ranges"))}
            if not rngs:
                w(f"      {n:<30} {str(op):<7} -")
                continue
            first = True
            for k, v in rngs.items():
                w(f"      {n if first else '':<30} {str(op) if first else '':<7} {k}={_fmt(v, 42)}")
                first = False

    # --- PPO ---
    sec("PPO / mạng")
    for f in dataclasses.fields(agent):
        v = getattr(agent, f.name)
        if dataclasses.is_dataclass(v):
            kv(f"agent.{f.name}", type(v).__name__)
            for sf in dataclasses.fields(v):
                w(f"      .{sf.name:<32} {_fmt(getattr(v, sf.name), 34)}")
        else:
            note = "<- GHI ĐÈ" if f.name == "max_iterations" else ""
            kv(f"agent.{f.name}", args.iters if f.name == "max_iterations" else v, note)

    w("")
    w(BAR)
    text = "\n".join(lines)
    print(text, flush=True)
    return text


# --------------------------------------------------------------------------- #
# Theo dõi VRAM
# --------------------------------------------------------------------------- #
class VramSampler(threading.Thread):
    """Lấy mẫu VRAM bằng nvidia-smi trong lúc train chạy."""

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None

    def run(self):
        if not self.available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                rows = out.stdout.strip().splitlines()
                if rows:
                    # Nhiều GPU: lấy card đang dùng nhiều nhất, vì OOM xảy ra
                    # theo TỪNG card chứ không theo tổng.
                    self.samples.append(max(int(r.strip()) for r in rows if r.strip()))
            except Exception:  # noqa: BLE001, S110
                pass
            self._stop.wait(self.interval)

    def stop(self) -> int | None:
        self._stop.set()
        self.join(timeout=8)
        return max(self.samples) if self.samples else None


# --------------------------------------------------------------------------- #
# Đọc metric
# --------------------------------------------------------------------------- #
def read_tfevents(run_dir: Path) -> dict:
    """Đọc Perf/total_fps + Train/mean_episode_length + Train/mean_reward."""
    out: dict = {"log_dir": str(run_dir)}
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except ImportError:
        out["error"] = "chưa cài tensorboard → không đọc được metric"
        return out

    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        out["error"] = "không có file tfevents"
        return out

    acc = ea.EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]

    if "Perf/total_fps" in tags:
        vals = [s.value for s in acc.Scalars("Perf/total_fps")]
        steady = vals[len(vals) // 2 :] or vals  # bỏ giai đoạn khởi động
        out["fps"] = int(statistics.median(steady))
    if "Train/mean_episode_length" in tags:
        s = acc.Scalars("Train/mean_episode_length")
        v = [x.value for x in s]
        out["iters_done"] = s[-1].step
        n = max(1, len(v) // 10)
        out["ep_len_first"] = round(statistics.mean(v[:n]), 1)
        out["ep_len_last"] = round(statistics.mean(v[-n:]), 1)
        out["ep_len_max"] = round(max(v), 1)
        out["ep_len_curve"] = [
            (s[i].step, round(s[i].value, 1))
            for i in (0, len(v) // 8, len(v) // 4, len(v) // 2, 3 * len(v) // 4, len(v) - 1)
        ]
    if "Train/mean_reward" in tags:
        v = [x.value for x in acc.Scalars("Train/mean_reward")]
        n = max(1, len(v) // 10)
        out["reward_first"] = round(statistics.mean(v[:n]), 2)
        out["reward_last"] = round(statistics.mean(v[-n:]), 2)
    return out


def count_gpus(gpu_ids: str | None) -> int:
    """Số GPU thực sự dùng, suy từ chuỗi --gpu-ids ('[0, 1]', 'all', '0'...).

    Cần con số này vì mỗi rank dựng ĐỦ num_envs riêng, nên tổng env và
    collection_size đều nhân với số GPU (rsl_rl/utils/logger.py:152).
    """
    if gpu_ids is None:
        return 1  # train.py mặc định gpu_ids=[0]
    if "all" in gpu_ids.lower():
        try:
            import torch

            return max(1, torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            return 1
    return max(1, len(re.findall(r"\d+", gpu_ids)))


def launch(
    task: str, num_envs: int, iters: int, exp_dir: Path,
    gpu_ids: str | None = None, extra: list[str] | None = None,
) -> tuple[int, Path | None, int | None, float]:
    """Chạy scripts/train.py, chỉ ghi đè num_envs + max_iterations (+ gpu_ids).

    KHÔNG capture stdout để tiến trình train hiện thẳng trên màn hình.
    ``extra`` để thêm cờ cho lần hiệu chỉnh (tách log, tắt W&B).
    Trả về (returncode, thư_mục_log_mới, VRAM_đỉnh_MB, wall_giây).
    """
    before = set(exp_dir.glob("*")) if exp_dir.exists() else set()
    cmd = [
        sys.executable, str(REPO / "scripts" / "train.py"), task,
        f"--env.scene.num-envs={num_envs}",
        f"--agent.max-iterations={iters}",
    ]
    if gpu_ids:
        # Truyền nguyên văn: tyro nhận '[0, 1]' hoặc 'all'.
        cmd += ["--gpu-ids", gpu_ids]
    cmd += list(extra or [])
    print("\n  $ " + " ".join(cmd) + "\n", flush=True)

    vram = VramSampler()
    vram.start()
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=REPO, check=False).returncode
    except KeyboardInterrupt:
        rc = 130
        print("\n  [Ctrl-C] đã dừng. Lưu ý: train.py KHÔNG lưu checkpoint khi bị ngắt.")
    wall = time.time() - t0
    peak = vram.stop()

    after = set(exp_dir.glob("*")) if exp_dir.exists() else set()
    new = sorted(after - before)
    return rc, (new[-1] if new else None), peak, wall


def latest_run(experiment: str | None = None) -> Path | None:
    pattern = f"{experiment}/*" if experiment else "*/*"
    runs = [p for p in LOG_ROOT.glob(pattern) if p.is_dir() and any(p.glob("events.out.tfevents.*"))]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# --------------------------------------------------------------------------- #
# Báo cáo
# --------------------------------------------------------------------------- #
def report(
    m: dict, step_dt: float, max_steps: int, num_envs: int, target_iters: int,
    n_gpu: int = 1, steps_per_env: int = 32,
) -> None:
    print("\n" + BAR)
    print("  KẾT QUẢ GIAI ĐOẠN 0")
    print(BAR)
    if "error" in m:
        print(f"  Không đọc được metric: {m['error']}")
        return

    print(f"  log_dir            {m['log_dir']}")
    print(f"  iterations đã chạy {m.get('iters_done', '?')} / {target_iters}")
    if n_gpu > 1:
        print(f"  GPU                {n_gpu} card × {num_envs:,} env = {n_gpu * num_envs:,} env tổng")
    if m.get("peak_vram_mb"):
        print(f"  VRAM đỉnh/card     {m['peak_vram_mb']:,} MB")

    per_iter = steps_per_env * num_envs * n_gpu
    fps = m.get("fps")
    print("\n  [1] THÔNG LƯỢNG")
    if fps:
        print(f"      Perf/total_fps          {fps:,}" + ("  (đã gộp mọi GPU)" if n_gpu > 1 else ""))
        print(f"      so với run cũ (34 FPS)  {fps / 34:,.0f}×")
        for it in (1000, 3000, 20001):
            secs = it * per_iter / fps
            print(f"      {it:>5} iter            ≈ {secs / 3600:6.2f} giờ")
        verdict = (
            "phần cứng là nút thắt — cân nhắc GPU mạnh hơn" if fps < 5000
            else "đủ dùng, đi tiếp Giai đoạn 1" if fps < 20000
            else "tốt"
        )
        print(f"      → {verdict}")
    else:
        print("      (không có Perf/total_fps)")

    print("\n  [2] TÍN HIỆU HỌC  — mean_episode_length")
    if "ep_len_first" in m:
        print(f"      trần episode            {max_steps} bước ({max_steps * step_dt:.0f} s)")
        print("      đường cong (iter: giá trị)")
        for step, val in m["ep_len_curve"]:
            bar = "█" * max(1, int(val / max_steps * 46))
            print(f"        {step:>6}: {val:>7.1f}  {bar}")
        delta = m["ep_len_last"] - m["ep_len_first"]
        print(f"      đầu → cuối              {m['ep_len_first']} → {m['ep_len_last']}  (Δ {delta:+.1f})")
        print(f"      đỉnh                    {m['ep_len_max']}")
        print(f"      reward                  {m.get('reward_first')} → {m.get('reward_last')}")

        if m["ep_len_last"] >= 300 and delta > 0:
            v = "ĐANG HỌC. Config chạy được ở quy mô này → sang Giai đoạn 1."
        elif delta > 20:
            v = "có tiến triển nhưng chậm. Chạy dài hơn, hoặc siết curriculum/DR."
        elif m["ep_len_last"] < 30:
            v = ("LẶP LẠI THẤT BẠI CŨ (ngã ngay). Quy mô không phải nguyên nhân "
                 "→ nghi vấn ở reward/termination/init, không phải num_envs.")
        else:
            v = "phẳng. Chưa học được gì đáng kể."
        print(f"      → {v}")
    else:
        print("      (không có Train/mean_episode_length)")
    print(BAR)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Giai đoạn 0 — đo throughput & tín hiệu học")
    p.add_argument("--task", default="VR-M3-1-Stand", help="mặc định: học đứng vững, full-body")
    p.add_argument("--num-envs", type=int, default=4096, help="env MỖI GPU, không phải tổng")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument(
        "--gpu-ids", default=None,
        help="truyền thẳng cho train.py, ví dụ '[0, 1]' hoặc 'all'. "
             "Bỏ trống = train.py dùng mặc định [0] (một GPU).",
    )
    p.add_argument(
        "--minutes", type=float, default=None,
        help="ngân sách train theo PHÚT. Chạy hiệu chỉnh ngắn để đo FPS rồi "
             "tính --iters cho vừa ngân sách (không tính thời gian khởi động).",
    )
    p.add_argument("--calib-iters", type=int, default=15, help="số iteration cho lần hiệu chỉnh")
    p.add_argument("--dry-run", action="store_true", help="chỉ in thông số rồi thoát")
    p.add_argument(
        "--analyze", nargs="?", const="latest", default=None,
        help="đọc lại run đã/đang chạy ('latest' hoặc đường dẫn log_dir), không train",
    )
    args = p.parse_args()

    sys.path.insert(0, str(REPO))
    import mjlab.tasks  # noqa: F401  (nạp registry)
    import src.tasks  # noqa: F401  (nạp registry)
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg(args.task)
    agent = load_rl_cfg(args.task)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    max_steps = int(cfg.episode_length_s / step_dt)
    n_gpu = count_gpus(args.gpu_ids)

    # --- chế độ chỉ phân tích ---
    if args.analyze:
        run_dir = latest_run(agent.experiment_name) if args.analyze == "latest" else Path(args.analyze)
        if run_dir is None or not run_dir.exists():
            print(f"Không tìm thấy run để phân tích: {args.analyze}")
            sys.exit(1)
        report(
            read_tfevents(run_dir), step_dt, max_steps, args.num_envs, args.iters,
            n_gpu, agent.num_steps_per_env,
        )
        return

    banner = print_run_params(args.task, args, cfg, agent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (OUT_DIR / f"params_{stamp}.txt").write_text(banner, encoding="utf-8")
    print(f"\n  Đã lưu thông số: logs/phase0/params_{stamp}.txt")

    if args.dry_run:
        print("  --dry-run: thoát, không chạy.")
        return

    exp_dir = LOG_ROOT / agent.experiment_name

    # --- hiệu chỉnh ngân sách phút -> số iteration ------------------------- #
    iters = args.iters
    if args.minutes:
        print("\n" + BAR)
        print(f"  HIỆU CHỈNH — chạy {args.calib_iters} iteration để đo FPS")
        print("  (logger=tensorboard + experiment riêng: KHÔNG tạo run W&B rác)")
        print(BAR, flush=True)
        calib_exp = f"{agent.experiment_name}_calib"
        calib_dir = LOG_ROOT / calib_exp
        c_rc, c_dir, _, c_wall = launch(
            args.task, args.num_envs, args.calib_iters, calib_dir, args.gpu_ids,
            extra=["--agent.logger=tensorboard", f"--agent.experiment-name={calib_exp}"],
        )
        if c_rc != 0 or c_dir is None:
            print(f"  Hiệu chỉnh thất bại (rc={c_rc}). Dừng.")
            sys.exit(1)
        c_fps = read_tfevents(c_dir).get("fps")
        if not c_fps:
            print("  Không đọc được FPS khi hiệu chỉnh. Dừng.")
            sys.exit(1)
        # Mỗi rank dựng đủ num_envs riêng ⇒ nhân thêm số GPU.
        per_iter = agent.num_steps_per_env * args.num_envs * n_gpu
        iters = max(1, int(args.minutes * 60 * c_fps / per_iter))
        print(f"\n  FPS đo được          {c_fps:,}")
        print(f"  ngân sách            {args.minutes} phút")
        print(f"  → max_iterations     {iters:,}")
        print(f"  (khởi động lần hiệu chỉnh mất {c_wall / 60:.1f} phút, không tính vào ngân sách)")

    # --- chạy train.py, chỉ ghi đè num_envs + max_iterations --------------- #
    before = set(exp_dir.glob("*")) if exp_dir.exists() else set()
    rc, run_dir, peak, wall = launch(args.task, args.num_envs, iters, exp_dir, args.gpu_ids)
    args.iters = iters  # để report() so đúng mốc
    print(f"\n  Kết thúc: rc={rc}, wall={wall / 60:.1f} phút")
    if run_dir is None:
        after = set(exp_dir.glob("*")) if exp_dir.exists() else set()
        new = sorted(after - before)
        run_dir = new[-1] if new else latest_run(agent.experiment_name)
    if run_dir is None:
        print("  Không tìm thấy thư mục log để phân tích.")
        sys.exit(1 if rc else 0)

    m = read_tfevents(run_dir)
    m["peak_vram_mb"] = peak
    m["wall_min"] = round(wall / 60, 1)
    m["returncode"] = rc
    m["n_gpu"] = n_gpu
    m["num_envs_per_gpu"] = args.num_envs
    report(m, step_dt, max_steps, args.num_envs, args.iters, n_gpu, agent.num_steps_per_env)

    out = OUT_DIR / f"results_{stamp}.json"
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Kết quả chi tiết: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
