#!/usr/bin/env python3
"""
CLI: parse natural language with LLM and publish /move_base_simple/goal (PoseStamped).
"""
import argparse
import json
import os
import sys
import importlib.util
from typing import Optional
from pathlib import Path

import rospy
from geometry_msgs.msg import PoseStamped


def _repo_root() -> Path:
    # .../src/planner/plan_manage/scripts/llm_goal_cli.py -> repo root
    return Path(__file__).resolve().parents[4]


def _load_landmarks(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Landmark file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(name: str) -> str:
    """标准化地标名称：去除空格、转小写"""
    return name.strip().lower().replace(" ", "").replace("_", "")


def _resolve_landmark(landmarks: dict, name: str) -> dict:
    """
    解析地标名称，支持：
    1. 精确匹配
    2. 大小写/空格容错
    3. 别名匹配（配置中的 "aliases" 字段）
    4. 模糊匹配（包含关系）
    """
    if not name:
        raise KeyError("Landmark name is empty")
    
    # 1. 精确匹配
    if name in landmarks:
        return landmarks[name]
    
    # 2. 标准化后匹配
    normalized_input = _normalize_name(name)
    for landmark_name, landmark_data in landmarks.items():
        if _normalize_name(landmark_name) == normalized_input:
            return landmark_data
        
        # 3. 别名匹配
        aliases = landmark_data.get("aliases", [])
        for alias in aliases:
            if _normalize_name(alias) == normalized_input:
                return landmark_data
    
    # 4. 模糊匹配（输入包含地标名或地标名包含输入）
    for landmark_name, landmark_data in landmarks.items():
        norm_landmark = _normalize_name(landmark_name)
        if normalized_input in norm_landmark or norm_landmark in normalized_input:
            print(f"[Fuzzy match] '{name}' -> '{landmark_name}'")
            return landmark_data
        
        # 也检查别名的模糊匹配
        for alias in landmark_data.get("aliases", []):
            norm_alias = _normalize_name(alias)
            if normalized_input in norm_alias or norm_alias in normalized_input:
                print(f"[Fuzzy match via alias] '{name}' -> '{landmark_name}' (alias: {alias})")
                return landmark_data
    
    raise KeyError(f"Landmark not configured: {name}")


# 地图边界配置（与 launch 文件中的 map_size 保持一致）
MAP_BOUNDS = {
    "x_min": -20.0, "x_max": 20.0,
    "y_min": -20.0, "y_max": 20.0,
    "z_min": 0.1,   "z_max": 3.0,
}


def _validate_coordinates(x: float, y: float, z: float) -> tuple:
    """
    验证坐标是否在安全范围内。
    返回 (is_valid, error_message, clamped_coords)
    """
    errors = []
    clamped_x, clamped_y, clamped_z = x, y, z
    
    # 检查 x 坐标
    if x < MAP_BOUNDS["x_min"] or x > MAP_BOUNDS["x_max"]:
        errors.append(f"x={x:.2f} out of range [{MAP_BOUNDS['x_min']}, {MAP_BOUNDS['x_max']}]")
        clamped_x = max(MAP_BOUNDS["x_min"], min(MAP_BOUNDS["x_max"], x))
    
    # 检查 y 坐标
    if y < MAP_BOUNDS["y_min"] or y > MAP_BOUNDS["y_max"]:
        errors.append(f"y={y:.2f} out of range [{MAP_BOUNDS['y_min']}, {MAP_BOUNDS['y_max']}]")
        clamped_y = max(MAP_BOUNDS["y_min"], min(MAP_BOUNDS["y_max"], y))
    
    # 检查 z 坐标（高度）
    if z < MAP_BOUNDS["z_min"] or z > MAP_BOUNDS["z_max"]:
        errors.append(f"z={z:.2f} out of range [{MAP_BOUNDS['z_min']}, {MAP_BOUNDS['z_max']}]")
        clamped_z = max(MAP_BOUNDS["z_min"], min(MAP_BOUNDS["z_max"], z))
    
    is_valid = len(errors) == 0
    error_msg = "; ".join(errors) if errors else None
    
    return is_valid, error_msg, (clamped_x, clamped_y, clamped_z)


def _build_pose_stamped(x: float, y: float, z: float, yaw: float = 0.0) -> PoseStamped:
    """构建 PoseStamped 消息，支持 yaw 角度"""
    import math
    
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    
    # 将 yaw 角度转换为四元数
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def _find_semantic_py(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"semantic.py not found at: {p}")

    candidates = [
        Path(__file__).resolve().parent / "semantic.py",
        _repo_root() / "semantic.py",
        Path.cwd() / "semantic.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "semantic.py not found. Use --semantic-path to specify its location."
    )


def _load_semantic_module(semantic_path: Path):
    spec = importlib.util.spec_from_file_location("semantic", str(semantic_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load semantic module from {semantic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    default_landmarks = _repo_root() / "src" / "planner" / "plan_manage" / "config" / "llm_landmarks.json"
    parser = argparse.ArgumentParser(
        description="LLM->Ego Planner CLI (/move_base_simple/goal)"
    )
    parser.add_argument("text", help='Natural language command, e.g. "take me to charger"')
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.2:1b-instruct-q4_0")
    parser.add_argument("--landmarks", default=str(default_landmarks))
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not publish")
    parser.add_argument("--override-x", type=float, default=None)
    parser.add_argument("--override-y", type=float, default=None)
    parser.add_argument("--override-z", type=float, default=None)
    parser.add_argument("--semantic-path", default=None, help="Path to semantic.py")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Load semantic.py from file path to avoid import path issues
    semantic_path = _find_semantic_py(args.semantic_path)
    semantic_module = _load_semantic_module(semantic_path)
    SemanticNavigator = semantic_module.SemanticNavigator

    landmarks = _load_landmarks(Path(args.landmarks))

    nav = SemanticNavigator(ollama_url=args.ollama_url)
    nav.model = args.model
    result = nav.parse_navigation_intent(args.text)

    action = result.get("action")
    landmark = result.get("landmark")
    confidence = float(result.get("confidence", 0.0))

    print("LLM result:", json.dumps(result, ensure_ascii=False, indent=2))

    if action != "navigate":
        print(f"Action is not navigate (action={action}); skip publishing.")
        return 1
    if confidence < args.min_confidence:
        print(f"Confidence too low: {confidence:.2f} < {args.min_confidence:.2f}; skip publishing.")
        return 1
    if not landmark:
        print("No landmark parsed; skip publishing.")
        return 1

    if args.override_x is not None and args.override_y is not None and args.override_z is not None:
        x, y, z = args.override_x, args.override_y, args.override_z
        yaw = 0.0
        print(f"Using override coordinates: ({x}, {y}, {z})")
    else:
        try:
            target = _resolve_landmark(landmarks, landmark)
        except KeyError as e:
            print(f"Error: {e}")
            print(f"Available landmarks: {list(landmarks.keys())}")
            return 1
        x = float(target["x"])
        y = float(target["y"])
        z = float(target.get("z", 1.0))
        yaw = float(target.get("yaw", 0.0))
        print(f"Resolved landmark '{landmark}' -> ({x}, {y}, {z})")

    # 验证坐标范围
    is_valid, error_msg, (clamped_x, clamped_y, clamped_z) = _validate_coordinates(x, y, z)
    if not is_valid:
        print(f"Warning: Coordinates out of bounds: {error_msg}")
        print(f"Clamping to safe range: ({clamped_x:.2f}, {clamped_y:.2f}, {clamped_z:.2f})")
        x, y, z = clamped_x, clamped_y, clamped_z

    if args.dry_run:
        print(f"[dry-run] target: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
        return 0

    rospy.init_node("llm_goal_cli", anonymous=True)
    pose_msg = _build_pose_stamped(x, y, z, yaw=yaw)
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    rospy.sleep(0.3)  # 等待 publisher 建立连接
    pub.publish(pose_msg)
    rospy.sleep(0.2)
    print(f"Published /move_base_simple/goal: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
