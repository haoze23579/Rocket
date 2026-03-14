"""
录制3D自动捕捉演示视频
自动运行仿真，逐帧截取OpenGL画面，输出MP4视频
"""
import numpy as np
import pygame
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
import cv2
import yaml
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env_wave import WaveSpaceEnvironment
from prediction.lstm_predictor import PhysicsPredictor, MotionPredictor
from planning.mpc_planner import MPCPlanner
from planning.grasp_strategy import GraspStrategy
from visualization.renderer_3d import SpaceRenderer3D


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_predictor(config, mode="lstm"):
    if mode == "reactive":
        return None
    if mode == "physics":
        print("[INFO] Using physics predictor (forced by mode)")
        return PhysicsPredictor(
            prediction_horizon=config["prediction"]["prediction_horizon"],
            dt=config["simulation"]["dt"]
        )

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "lstm_predictor.pth")
    norm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "normalization.npz")
    if os.path.exists(model_path) and os.path.exists(norm_path):
        try:
            pred_cfg = config["prediction"]
            predictor = MotionPredictor(
                model_path=model_path,
                hidden_size=pred_cfg["hidden_size"],
                num_layers=pred_cfg["num_layers"],
                prediction_horizon=pred_cfg["prediction_horizon"]
            )
            nd = np.load(norm_path)
            predictor.set_normalization(nd["mean"], nd["std"])
            print("[INFO] Loaded LSTM predictor")
            return predictor
        except Exception as e:
            print(f"[WARN] LSTM failed: {e}")
    print("[INFO] Using physics predictor")
    return PhysicsPredictor(
        prediction_horizon=config["prediction"]["prediction_horizon"],
        dt=config["simulation"]["dt"]
    )


def capture_frame(width, height):
    """从OpenGL帧缓冲截取一帧"""
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    frame = np.flipud(frame)       # OpenGL是底部起始
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV用BGR
    return frame


def main(args):
    config = load_config()
    width = config["display"]["width"]
    height = config["display"]["height"]

    env = WaveSpaceEnvironment(config)
    renderer = SpaceRenderer3D(width=width, height=height, star_count=2000)
    predictor = load_predictor(config, mode=args.mode)
    planner = MPCPlanner(env.arm, config)
    strategy = GraspStrategy(config)

    # 视频输出
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"demo_recording_{args.mode}.mp4"
    )
    video_fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    if not writer.isOpened():
        print("[ERROR] Failed to open video writer!")
        print("Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    print(f"[INFO] Recording to: {output_path}")
    print(f"[INFO] Resolution: {width}x{height} @ {video_fps}fps")

    # 波次配置 - 录制用，碎片数量适中
    wave_cfg = {"count": 4, "min_speed": 1.5, "max_speed": 3.5, "interval": 5.0}
    env.reset(wave_cfg)
    planner.reset()
    renderer.clear_trails()

    state = "WAITING"
    predicted_trajectory = None
    predicted_confidence = None

    # 仿真参数
    sim_dt = config["simulation"]["dt"]
    render_interval = int(1.0 / (video_fps * sim_dt))
    render_interval = max(1, int(render_interval * max(1, int(args.sim_speedup))))  # 每隔多少仿真步渲染一帧
    max_sim_time = float(args.max_sim_time)  # 最长录制60秒
    frame_count = 0
    sim_step = 0

    # 相机缓慢旋转
    cam_auto_rotate = True

    start_time = time.time()
    print(f"[INFO] Recording started... mode={args.mode}, sim_speedup={args.sim_speedup}x (press ESC to stop early)")

    running = True
    while running and env.time < max_sim_time and not env.done:
        # 事件处理（允许ESC提前退出）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # ---- 仿真逻辑 ----
        info = env._get_info()
        distance = info["distance"]
        has_debris = env.current_debris is not None

        target_pos = None
        close_gripper = False

        if not has_debris:
            state = "WAITING"
            target_pos = np.array([2.0, 1.0, 0.0])
            predicted_trajectory = None
        elif env.holding_debris:
            state = "RETURNING"
            target_pos = env.recycle_zone.position.copy()
            target_pos[1] = 0.5
            close_gripper = True
            predicted_trajectory = None
        else:
            obs = env.sensor.observe(env.current_debris, env.time)
            obs_seq = env.sensor.get_observation_sequence(
                config["prediction"]["sequence_length"]
            )
            if obs_seq is None:
                state = "WAITING"
            else:
                if args.mode == "reactive":
                    predicted_trajectory = None
                    predicted_confidence = None
                    target_pos = obs_seq[-1, :3].copy()
                else:
                    predicted_trajectory, predicted_confidence = \
                        predictor.predict_position_only(obs_seq)
                    target_pos = planner.plan(
                        predicted_trajectory, confidence=predicted_confidence
                    )
                state = "APPROACHING" if distance < 3.0 else "TRACKING"

                if distance < env.grasp_dist_threshold * 2:
                    tip_vel = env.arm.get_tip_velocity()
                    rel_vel = np.linalg.norm(tip_vel - env.current_debris.velocity)
                    if distance < env.grasp_dist_threshold and rel_vel < env.grasp_vel_threshold:
                        state = "GRASPING"
                        close_gripper = True

        info, wave_done = env.step(target_pos, close_gripper)

        # 轨迹 & 粒子
        if env.current_debris:
            renderer.update_trail(env.current_debris.position, env.arm.get_tip_position())
        if target_pos is not None:
            tip = env.arm.get_tip_position()
            vel = env.arm.get_tip_velocity()
            if np.linalg.norm(vel) > 0.1:
                renderer.thruster_particles.emit(
                    tip, -vel * 0.3, color=(1.0, 0.6, 0.1, 0.8), count=2, spread=0.15
                )
        renderer.thruster_particles.update(sim_dt)

        for evt in env.events:
            if evt["time"] >= env.time - sim_dt:
                tip = env.arm.get_tip_position()
                if evt["type"] == "recycled":
                    renderer.spark_particles.emit(
                        tip, np.zeros(3), color=(0.2, 1.0, 0.4, 1.0),
                        count=60, spread=1.5, lifetime=2.0
                    )
                    renderer.clear_trails()
                elif evt["type"] == "captured":
                    renderer.spark_particles.emit(
                        tip, np.zeros(3), color=(1.0, 1.0, 0.3, 1.0),
                        count=30, spread=0.8, lifetime=1.0
                    )

        if wave_done:
            state = "DONE"

        sim_step += 1

        # ---- 渲染 & 录制 (每隔render_interval步渲染一帧) ----
        if sim_step % render_interval == 0:
            # 相机
            if env.current_debris:
                mid = (env.arm.get_tip_position() + env.current_debris.position) / 2
                renderer.cam_target = mid
            else:
                renderer.cam_target = env.arm.get_tip_position()

            if cam_auto_rotate:
                renderer.cam_yaw += 0.15

            renderer.begin_frame()
            renderer.draw_stars()
            renderer.draw_grid()
            renderer.draw_axes()

            # 回收区
            renderer.draw_recycle_zone(
                env.recycle_zone.position, env.recycle_zone.radius,
                env.recycle_zone.collected_count, pulse_time=env.time
            )

            # 碎片 + 高速拖尾
            all_debris = env.get_all_debris_positions()
            for pos, shape, dstate, orient, vel, speed in all_debris:
                if dstate == "warning":
                    renderer.draw_warning_indicator(pos, speed)
                else:
                    trail_color = (1.0, 0.4, 0.2) if speed > 3.0 else (1.0, 0.7, 0.3)
                    renderer.draw_speed_trail(pos, vel, trail_color)
                    renderer.draw_debris(pos, orient, np.array([2.0, 1.5, 1.0]), shape)

            # 轨迹
            renderer.draw_trail(renderer.debris_trail, (1.0, 0.4, 0.4))
            renderer.draw_trail(renderer.arm_trail, (0.4, 0.7, 1.0))

            # 预测轨迹
            if predicted_trajectory is not None:
                renderer.draw_predicted_trajectory(predicted_trajectory, predicted_confidence)

            # 机械臂
            joint_positions = env.arm.get_joint_positions()
            renderer.draw_robot_arm(
                joint_positions, env.arm.gripper_open,
                highlight=(state in ("GRASPING", "RETURNING"))
            )

            # 连接线
            if env.current_debris and not env.holding_debris:
                renderer.draw_connection_line(
                    env.arm.get_tip_position(), env.current_debris.position
                )
            if env.holding_debris:
                renderer.draw_connection_line(
                    env.arm.get_tip_position(), env.recycle_zone.position
                )

            # 粒子
            renderer.thruster_particles.draw()
            renderer.spark_particles.draw()
            renderer.spark_particles.update(1.0 / video_fps)

            # HUD
            renderer.draw_wave_hud(info, mode="AUTO")
            renderer.draw_state_label(state)

            renderer.end_frame(0)  # 不限帧率，尽快渲染

            # 截帧写入视频
            frame = capture_frame(width, height)
            writer.write(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  Frame {frame_count} | Sim time: {env.time:.1f}s | "
                      f"Real time: {elapsed:.1f}s | State: {state} | "
                      f"Recycled: {env.wave.recycled_count}/{env.wave.total}")

    # 波次完成后多录几秒结果画面
    if env.done:
        print("[INFO] Wave complete, recording result screen...")
        for _ in range(video_fps * 3):  # 3秒结果画面
            for event in pygame.event.get():
                pass

            renderer.begin_frame()
            renderer.draw_stars()
            renderer.draw_grid()
            renderer.draw_recycle_zone(
                env.recycle_zone.position, env.recycle_zone.radius,
                env.recycle_zone.collected_count, pulse_time=env.time
            )
            joint_positions = env.arm.get_joint_positions()
            renderer.draw_robot_arm(joint_positions, env.arm.gripper_open)
            renderer.spark_particles.draw()
            renderer.spark_particles.update(1.0 / video_fps)

            # 结果覆盖
            renderer.begin_2d_overlay()
            from OpenGL.GL import glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS
            glColor4f(0, 0, 0, 0.6)
            glBegin(GL_QUADS)
            glVertex2f(0, 0); glVertex2f(width, 0)
            glVertex2f(width, height); glVertex2f(0, height)
            glEnd()

            cx, cy = width // 2, height // 2
            total = env.wave.total
            recycled = env.wave.recycled_count
            missed = env.wave.missed_count
            rate = recycled / max(1, total) * 100

            renderer._draw_text_2d(cx - 140, cy - 80, "WAVE COMPLETE",
                                    renderer.hud_font_title, (60, 220, 255))
            renderer._draw_text_2d(cx - 120, cy - 30,
                                    f"Recycled: {recycled}/{total} ({rate:.0f}%)",
                                    renderer.hud_font_large, (100, 255, 100))
            renderer._draw_text_2d(cx - 80, cy + 10,
                                    f"Missed: {missed}",
                                    renderer.hud_font_large, (255, 100, 100))
            renderer._draw_text_2d(cx - 80, cy + 50,
                                    f"Time: {env.time:.1f}s",
                                    renderer.hud_font_large, (220, 220, 220))
            renderer.end_2d_overlay()

            renderer.end_frame(0)
            frame = capture_frame(width, height)
            writer.write(frame)
            frame_count += 1

    # 收尾
    writer.release()
    renderer.close()

    elapsed = time.time() - start_time
    duration = frame_count / video_fps
    print(f"\n{'='*50}")
    print(f"  Recording complete!")
    print(f"  Output: {output_path}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Real time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record demo video by experiment mode")
    parser.add_argument("--mode", choices=["lstm", "physics", "reactive"], default="lstm",
                        help="recording mode")
    parser.add_argument("--sim-speedup", type=int, default=4,
                        help="speedup factor by frame skipping")
    parser.add_argument("--max-sim-time", type=float, default=60.0,
                        help="maximum simulated seconds to record")
    parser.add_argument("--output", type=str, default="",
                        help="output path, default demo_recording_<mode>.mp4")
    cli_args = parser.parse_args()
    main(cli_args)
