#!/usr/bin/env python3
"""Render retargeted robot CSV motion to an offscreen MP4 video.

The motion CSV convention is the same as ``multi_robot_visualize.py``:
position XYZ followed by quaternion XYZW and then the robot joint positions.
The renderer converts the quaternion to MuJoCo's WXYZ order before writing each
frame.  Multiple robots can be rendered together for a direct source/target
comparison.

Example:
    python scripts/render_motion_video.py \
        --motion walk_forward_loop_002__A022_from_g1 \
        --robots g1 booster_k1 \
        --source-fps 120 \
        --render-fps 30 \
        --output output_data/videos/g1_to_booster_k1.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from multi_robot_visualize import build_combined_spec, get_qpos_start, get_robot_xml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_DIR = PROJECT_ROOT / "output_data" / "robot_motion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        required=True,
        help="Motion stem without robot suffix, e.g. walk_forward_loop_002__A022_from_g1",
    )
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=4.2)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.75),
        metavar=("X", "Y", "Z"),
    )
    return parser.parse_args()


def load_motion(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Motion file must be a 2D qpos CSV with at least 7 columns: {path}")

    qpos = data.astype(np.float64, copy=True)
    quat_xyzw = data[:, 3:7].copy()
    qpos[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    return qpos


def build_robot_motion_map(motion_dir: Path, motion: str, robots: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for robot in robots:
        path = motion_dir / f"{motion}_{robot}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Motion file not found for {robot}: {path}")
        result[robot] = load_motion(path)
    return result


def robot_offsets(robots: list[str], spacing: float = 2.0) -> dict[str, tuple[float, float]]:
    columns = max(1, math.ceil(math.sqrt(len(robots))))
    offsets: dict[str, tuple[float, float]] = {}
    for index, robot in enumerate(robots):
        column = index % columns
        row = index // columns
        offsets[robot] = (
            (column - (columns - 1) / 2.0) * spacing,
            (row - (math.ceil(len(robots) / columns) - 1) / 2.0) * spacing,
        )
    return offsets


def render(args: argparse.Namespace) -> tuple[Path, int, float]:
    robots = list(args.robots)
    robot_qpos = build_robot_motion_map(args.motion_dir, args.motion, robots)
    n_frames = min(motion.shape[0] for motion in robot_qpos.values())
    if n_frames == 0:
        raise ValueError("No motion frames were loaded")
    if args.source_fps <= 0.0 or args.render_fps <= 0.0:
        raise ValueError("source-fps and render-fps must be positive")

    spec = build_combined_spec(robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    starts: dict[str, int] = {}
    for robot in robots:
        starts[robot] = get_qpos_start(model, f"{robot}_floating_base_joint")
        qpos_dim = robot_qpos[robot].shape[1]
        model_qpos_dim = mujoco.MjModel.from_xml_path(get_robot_xml(robot)).nq
        if qpos_dim != model_qpos_dim:
            raise ValueError(
                f"qpos dimension mismatch for {robot}: CSV has {qpos_dim} values, "
                f"model expects {model_qpos_dim}"
            )

    offsets = robot_offsets(robots)
    model.vis.global_.offwidth = int(args.width)
    model.vis.global_.offheight = int(args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.asarray(args.lookat, dtype=np.float64)
    camera.distance = float(args.distance)
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)

    step = max(1, round(args.source_fps / args.render_fps))
    actual_fps = args.source_fps / step
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    total = math.ceil(n_frames / step)
    with imageio.get_writer(
        str(output),
        fps=actual_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame_index in range(0, n_frames, step):
            for robot in robots:
                start = starts[robot]
                qpos = robot_qpos[robot][frame_index]
                data.qpos[start : start + qpos.shape[0]] = qpos
                dx, dy = offsets[robot]
                data.qpos[start] += dx
                data.qpos[start + 1] += dy

            root_xy = np.asarray(
                [data.qpos[starts[robot] : starts[robot] + 2] for robot in robots],
                dtype=np.float64,
            )
            camera.lookat[0:2] = root_xy.mean(axis=0)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

            if (frame_index // step + 1) == total or (frame_index // step + 1) % 25 == 0:
                print(f"rendered {frame_index // step + 1}/{total} frames", flush=True)

    renderer.close()
    return output, total, actual_fps


def main() -> None:
    args = parse_args()
    output, frames, fps = render(args)
    print(f"saved={output} frames={frames} fps={fps:g}")


if __name__ == "__main__":
    main()
