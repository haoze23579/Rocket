"""
批量实验脚本
在不同参数下运行仿真实验，生成性能报告
支持三组对比：LSTM预测 / 物理预测(线性外推) / 无预测(纯反应式)
"""
import numpy as np
import yaml
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env import SpaceEnvironment
from prediction.lstm_predictor import PhysicsPredictor, MotionPredictor
from planning.mpc_planner import MPCPlanner
from planning.grasp_strategy import GraspStrategy
from evaluation.metrics import ExperimentMetrics, TrialResult
from evaluation.report_generator import ReportGenerator


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_trial(env, predictor, strategy, config,
                     linear_speed, angular_speed, shape,
                     mode="lstm") -> TrialResult:
    """
    运行单次实验
    mode: "lstm" / "physics" / "reactive"
    """
    env.reset(linear_speed=linear_speed, angular_speed=angular_speed, shape=shape)

    state = "SEARCHING"
    max_steps = int(30.0 / config["simulation"]["dt"])
    prediction_errors = []
    cached_target = None
    pred_interval = 5  # LSTM每5步推理一次

    for step in range(max_steps):
        if env.done:
            break

        obs = env.sensor.observe(env.debris, env.time)
        obs_seq = env.sensor.get_observation_sequence(config["prediction"]["sequence_length"])

        target_pos = None
        close_gripper = False

        if state == "SEARCHING":
            if obs_seq is not None:
                state = "TRACKING"

        elif state in ("TRACKING", "APPROACHING"):
            if obs_seq is not None:
                observed_pos = obs_seq[-1, :3]
                info_check = env._get_info()
                dist = info_check["distance"]

                if mode == "lstm" and predictor is not None:
                    if cached_target is None or step % pred_interval == 0:
                        pred_pos, pred_conf = predictor.predict_position_only(obs_seq)

                        # 用LSTM预测的位移趋势做提前量
                        # 关键：pred_pos是相对预测，需要算偏移再加到观测位置
                        if dist > 5.0:
                            lead_steps = min(12, len(pred_pos) - 1)
                            lead_scale = 1.0
                        elif dist > 3.0:
                            lead_steps = min(8, len(pred_pos) - 1)
                            lead_scale = 0.7
                        elif dist > 1.5:
                            lead_steps = min(4, len(pred_pos) - 1)
                            lead_scale = 0.4
                        else:
                            lead_steps = min(2, len(pred_pos) - 1)
                            lead_scale = 0.2

                        if lead_steps > 0 and len(pred_pos) > lead_steps:
                            lead_offset = (pred_pos[lead_steps] - pred_pos[0]) * lead_scale
                            cached_target = observed_pos + lead_offset
                        else:
                            cached_target = observed_pos.copy()

                        # 记录预测误差
                        if len(pred_pos) > 0:
                            pred_error = np.linalg.norm(pred_pos[0] - env.debris.position)
                            prediction_errors.append(pred_error)

                    target_pos = cached_target

                elif mode == "physics":
                    # 物理预测：基于观测速度的线性外推
                    observed_vel = obs_seq[-1, 3:6]
                    if dist > 3.0:
                        lead_time = 0.08  # 80ms提前量
                    elif dist > 1.5:
                        lead_time = 0.05
                    else:
                        lead_time = 0.02
                    target_pos = observed_pos + observed_vel * lead_time

                else:
                    # 纯反应式：直接追踪观测位置
                    target_pos = observed_pos.copy()

            info = env._get_info()
            if info["distance"] < 3.0:
                state = "APPROACHING"

            # 抓取判断
            if info["distance"] < config["grasp"]["success_threshold"] * 2:
                tip_vel = env.arm.get_tip_velocity()
                rel_vel = np.linalg.norm(tip_vel - env.debris.velocity)
                gripper_type = strategy.select_gripper(shape, np.max(env.debris.size))
                should, _ = strategy.should_grasp(info["distance"], rel_vel, gripper_type)
                if should:
                    close_gripper = True

        info, done, success = env.step(target_pos, close_gripper)

    min_dist = min(env.capture_distance_history) if env.capture_distance_history else float('inf')
    tip_vel = env.arm.get_tip_velocity()
    rel_vel = np.linalg.norm(tip_vel - env.debris.velocity) if env.debris else 0

    return TrialResult(
        success=env.success,
        capture_time=env.time,
        min_distance=min_dist,
        relative_velocity_at_grasp=rel_vel,
        debris_linear_speed=linear_speed,
        debris_angular_speed=angular_speed,
        debris_shape=shape,
        gripper_type="three_finger",
        prediction_used=(mode != "reactive"),
        prediction_error=np.mean(prediction_errors) if prediction_errors else 0
    )


def run_experiment_group(config, mode="lstm", predictor=None):
    """运行一组实验"""
    env = SpaceEnvironment(config)
    strategy = GraspStrategy(config)
    metrics = ExperimentMetrics()

    # 速度网格：扩展到高速段，凸显LSTM预测优势
    speeds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ang_speeds = [0.0, 0.52, 1.05]
    shapes = config["debris"]["shapes"]
    trials_per_combo = 3

    label = {"lstm": "LSTM预测", "physics": "物理预测", "reactive": "纯反应式"}[mode]
    print(f"\n{'='*50}", flush=True)
    print(f"  实验组: {label} (mode={mode})", flush=True)
    print(f"{'='*50}", flush=True)

    trial_count = 0
    for speed in speeds:
        for ang_speed in ang_speeds:
            for rep in range(trials_per_combo):
                shape = np.random.choice(shapes)
                result = run_single_trial(
                    env, predictor, strategy, config,
                    speed, ang_speed, shape, mode
                )
                metrics.trials.append(result)
                trial_count += 1

                if trial_count % 5 == 0:
                    rate = metrics.success_rate * 100
                    print(f"  Trial {trial_count}: success={rate:.1f}% "
                          f"(v={speed:.1f}, ω={ang_speed:.2f})", flush=True)

    return metrics


def main():
    config = load_config()
    report_gen = ReportGenerator(
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    )

    # 加载LSTM模型
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "lstm_predictor.pth")
    pred_cfg = config["prediction"]
    lstm_predictor = None

    if os.path.exists(model_path):
        lstm_predictor = MotionPredictor(
            model_path=model_path,
            hidden_size=pred_cfg["hidden_size"],
            num_layers=pred_cfg["num_layers"],
            prediction_horizon=pred_cfg["prediction_horizon"]
        )
        norm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "normalization.npz")
        if os.path.exists(norm_path):
            norm_data = np.load(norm_path)
            lstm_predictor.set_normalization(norm_data["mean"], norm_data["std"])
        print("LSTM模型已加载", flush=True)
    else:
        print("警告: 未找到LSTM模型，跳过LSTM组", flush=True)

    start_time = time.time()

    # 三组实验
    metrics_lstm = run_experiment_group(config, "lstm", lstm_predictor) if lstm_predictor else None
    metrics_physics = run_experiment_group(config, "physics")
    metrics_reactive = run_experiment_group(config, "reactive")

    elapsed = time.time() - start_time
    print(f"\n总实验时间: {elapsed:.1f}s")

    # 生成报告（用LSTM组作为主报告，reactive作为对比）
    primary = metrics_lstm if metrics_lstm else metrics_physics
    report_gen.generate_full_report(primary, metrics_reactive)

    # 打印三组对比
    print("\n" + "="*70)
    print("  三组对比结果")
    print("="*70)
    groups = []
    if metrics_lstm:
        groups.append(("LSTM预测", metrics_lstm))
    groups.append(("物理预测", metrics_physics))
    groups.append(("纯反应式", metrics_reactive))

    print(f"{'指标':<25}", end="")
    for name, _ in groups:
        print(f"{name:>12}", end="")
    print()
    print("-" * (25 + 12 * len(groups)))

    for label, getter in [
        ("成功率", lambda m: f"{m.success_rate*100:.1f}%"),
        ("平均捕获时间(s)", lambda m: f"{m.avg_capture_time:.2f}"),
        ("平均相对速度(m/s)", lambda m: f"{m.avg_relative_velocity:.3f}"),
        ("平均最小距离(m)", lambda m: f"{m.avg_min_distance:.3f}"),
    ]:
        print(f"{label:<25}", end="")
        for name, m in groups:
            print(f"{getter(m):>12}", end="")
        print()

    # 按速度分组对比
    print("\n" + "="*70)
    print("  按碎片速度分组的成功率")
    print("="*70)
    all_speeds = sorted(set(t.debris_linear_speed for t in (metrics_lstm or metrics_physics).trials))
    print(f"{'速度(m/s)':<12}", end="")
    for name, _ in groups:
        print(f"{name:>12}", end="")
    print()
    print("-" * (12 + 12 * len(groups)))

    for speed in all_speeds:
        print(f"{speed:<12.1f}", end="")
        for name, m in groups:
            trials_at_speed = [t for t in m.trials if abs(t.debris_linear_speed - speed) < 0.01]
            if trials_at_speed:
                rate = sum(1 for t in trials_at_speed if t.success) / len(trials_at_speed) * 100
                print(f"{rate:>11.1f}%", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # 保存实验日志
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("太空垃圾捕获仿真 - 实验结果\n")
        f.write(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed:.1f}s\n\n")

        for name, m in groups:
            s = m.summary()
            f.write(f"[{name}]\n")
            f.write(f"  总试次: {s['total_trials']}\n")
            f.write(f"  成功率: {s['success_rate']*100:.1f}%\n")
            f.write(f"  平均捕获时间: {s['avg_capture_time']:.2f}s\n")
            f.write(f"  平均相对速度: {s['avg_relative_velocity']:.3f}m/s\n")
            f.write(f"  平均最小距离: {s['avg_min_distance']:.3f}m\n\n")

        f.write("按速度分组:\n")
        for speed in all_speeds:
            f.write(f"  v={speed:.1f} m/s: ")
            for name, m in groups:
                trials_at_speed = [t for t in m.trials if abs(t.debris_linear_speed - speed) < 0.01]
                if trials_at_speed:
                    rate = sum(1 for t in trials_at_speed if t.success) / len(trials_at_speed) * 100
                    f.write(f"{name}={rate:.0f}%  ")
            f.write("\n")

    print(f"\n实验日志已保存到 {log_path}")


if __name__ == "__main__":
    main()
