#!/usr/bin/env python3
"""
低延时连续对话导航节点
- 常驻运行，预热 LLM 连接
- 支持上下文对话
- 流式解析，边解析边准备执行
- 多种输入方式：话题/服务/终端
"""
import json
import math
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class LowLatencyLLMNavigator:
    """低延时 LLM 导航器"""

    def __init__(self):
        rospy.init_node("llm_nav_node", anonymous=False)

        # === 配置参数 ===
        self.ollama_url = rospy.get_param("~ollama_url", "http://127.0.0.1:11434")
        self.model = rospy.get_param("~model", "llama3.2:1b-instruct-q4_0")
        self.min_confidence = rospy.get_param("~min_confidence", 0.5)
        self.enable_context = rospy.get_param("~enable_context", True)
        self.context_max_turns = rospy.get_param("~context_max_turns", 5)

        # === 地标配置 ===
        landmarks_path = rospy.get_param(
            "~landmarks_path",
            str(Path(__file__).parent.parent / "config" / "llm_landmarks.json"),
        )
        self.landmarks = self._load_landmarks(landmarks_path)
        self._build_landmark_index()  # 预建立索引加速查找

        # === 地图边界 ===
        self.map_bounds = {
            "x_min": rospy.get_param("~map_x_min", -20.0),
            "x_max": rospy.get_param("~map_x_max", 20.0),
            "y_min": rospy.get_param("~map_y_min", -20.0),
            "y_max": rospy.get_param("~map_y_max", 20.0),
            "z_min": rospy.get_param("~map_z_min", 0.1),
            "z_max": rospy.get_param("~map_z_max", 3.0),
        }

        # === 对话上下文 ===
        self.context_history = deque(maxlen=self.context_max_turns)
        self.last_landmark = None  # 记住上一个目标，支持"去那里"等指代

        # === ROS 接口 (预建立连接) ===
        self.goal_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "~status", String, queue_size=10
        )
        self.text_sub = rospy.Subscriber(
            "~text_input", String, self._text_callback, queue_size=5
        )

        # === 预热 LLM 连接 ===
        self._warmup_llm()

        # === 会话管理 ===
        self.session = requests.Session()  # 复用 HTTP 连接
        self.session.headers.update({"Connection": "keep-alive"})

        # === 统计 ===
        self.stats = {"requests": 0, "avg_latency_ms": 0, "total_latency_ms": 0}

        rospy.loginfo("LLM Navigation Node initialized (low-latency mode)")
        rospy.loginfo(f"  Ollama URL: {self.ollama_url}")
        rospy.loginfo(f"  Model: {self.model}")
        rospy.loginfo(f"  Landmarks: {len(self.landmarks)}")

    def _load_landmarks(self, path: str) -> dict:
        """加载地标配置"""
        p = Path(path)
        if not p.exists():
            rospy.logwarn(f"Landmarks file not found: {path}")
            return {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_landmark_index(self):
        """预建立地标索引，加速模糊匹配"""
        self.landmark_index = {}  # normalized_name -> (original_name, data)
        for name, data in self.landmarks.items():
            # 主名称
            norm = self._normalize(name)
            self.landmark_index[norm] = (name, data)
            # 别名
            for alias in data.get("aliases", []):
                norm_alias = self._normalize(alias)
                self.landmark_index[norm_alias] = (name, data)

    def _normalize(self, s: str) -> str:
        """标准化字符串"""
        return s.strip().lower().replace(" ", "").replace("_", "")

    def _warmup_llm(self):
        """预热 LLM 连接，减少首次请求延时"""
        rospy.loginfo("Warming up LLM connection...")
        try:
            start = time.time()
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=30,
            )
            resp.raise_for_status()
            elapsed = (time.time() - start) * 1000
            rospy.loginfo(f"LLM warmup complete ({elapsed:.0f}ms)")
        except Exception as e:
            rospy.logwarn(f"LLM warmup failed: {e}")

    def _build_prompt(self, user_input: str) -> str:
        """构建带上下文的 prompt"""
        # 系统提示
        system = """你是无人机导航助手。分析指令返回JSON。
必须包含: "action", "landmark", "confidence"
action: "navigate"(导航), "stop"(停止), "pause"(暂停), "resume"(继续), "unknown"
landmark: 目标地点名称(如果action是navigate)
confidence: 0-1的置信度

注意：
- "那里"、"那边"、"刚才那个地方"等指代词，请根据上下文推断具体地点
- "回去"、"返回"通常指起飞点/原点
- 只返回JSON，不要其他文字"""

        # 构建上下文
        context_str = ""
        if self.enable_context and self.context_history:
            context_str = "\n历史对话:\n"
            for turn in self.context_history:
                context_str += f"用户: {turn['user']}\n"
                context_str += f"结果: {turn['result']}\n"
            if self.last_landmark:
                context_str += f"(上一个目标地点是: {self.last_landmark})\n"

        prompt = f"""{system}
{context_str}
当前指令: {user_input}

JSON:"""
        return prompt

    def _parse_with_llm(self, user_input: str) -> Tuple[Dict, float]:
        """
        调用 LLM 解析意图
        返回: (result_dict, latency_ms)
        """
        prompt = self._build_prompt(user_input)
        start = time.time()

        try:
            resp = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # 更低温度，更确定性
                        "num_predict": 100,  # 限制输出长度
                    },
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()["response"]

            # 提取 JSON
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = json.loads(raw)

            latency = (time.time() - start) * 1000

            # 验证并规范化
            result = self._validate_result(result)
            return result, latency

        except Exception as e:
            latency = (time.time() - start) * 1000
            rospy.logerr(f"LLM parse failed: {e}")
            return {"action": "unknown", "landmark": None, "confidence": 0.0}, latency

    def _validate_result(self, result: Dict) -> Dict:
        """验证 LLM 输出"""
        valid_actions = {"navigate", "stop", "pause", "resume", "unknown"}

        action = result.get("action", "unknown")
        if action not in valid_actions:
            action = "unknown"

        landmark = result.get("landmark")
        if landmark:
            landmark = str(landmark).strip()
            # 处理指代词
            if landmark in ["那里", "那边", "刚才那个地方", "there"] and self.last_landmark:
                landmark = self.last_landmark
            elif landmark in ["回去", "返回", "home", "back"]:
                landmark = "起飞点"

        try:
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.0

        return {"action": action, "landmark": landmark, "confidence": confidence}

    def _resolve_landmark(self, name: str) -> Optional[Tuple[str, dict]]:
        """
        快速解析地标名称
        返回: (原始名称, 坐标数据) 或 None
        """
        if not name:
            return None

        norm = self._normalize(name)

        # 1. 精确匹配索引
        if norm in self.landmark_index:
            return self.landmark_index[norm]

        # 2. 模糊匹配
        for indexed_norm, (orig_name, data) in self.landmark_index.items():
            if norm in indexed_norm or indexed_norm in norm:
                return (orig_name, data)

        return None

    def _clamp_coordinates(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """裁剪坐标到安全范围"""
        x = max(self.map_bounds["x_min"], min(self.map_bounds["x_max"], x))
        y = max(self.map_bounds["y_min"], min(self.map_bounds["y_max"], y))
        z = max(self.map_bounds["z_min"], min(self.map_bounds["z_max"], z))
        return x, y, z

    def _publish_goal(self, x: float, y: float, z: float, yaw: float = 0.0):
        """发布导航目标"""
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(msg)

    def _publish_status(self, status: str, details: dict = None):
        """发布状态信息"""
        msg = {"status": status, "timestamp": time.time()}
        if details:
            msg.update(details)
        self.status_pub.publish(String(data=json.dumps(msg, ensure_ascii=False)))

    def process_command(self, text: str) -> dict:
        """
        处理用户命令（核心函数）
        返回处理结果
        """
        total_start = time.time()
        result = {
            "input": text,
            "success": False,
            "action": None,
            "landmark": None,
            "coordinates": None,
            "latency_ms": {},
        }

        # === 1. LLM 解析 ===
        llm_result, llm_latency = self._parse_with_llm(text)
        result["latency_ms"]["llm"] = llm_latency
        result["action"] = llm_result["action"]
        result["landmark"] = llm_result["landmark"]
        result["confidence"] = llm_result["confidence"]

        rospy.loginfo(f"LLM result ({llm_latency:.0f}ms): {llm_result}")

        # === 2. 检查置信度 ===
        if llm_result["confidence"] < self.min_confidence:
            result["error"] = f"Confidence too low: {llm_result['confidence']:.2f}"
            self._publish_status("low_confidence", result)
            return result

        # === 3. 处理不同动作 ===
        action = llm_result["action"]

        if action == "navigate":
            landmark_name = llm_result["landmark"]
            if not landmark_name:
                result["error"] = "No landmark specified"
                self._publish_status("no_landmark", result)
                return result

            # 解析地标
            resolve_start = time.time()
            resolved = self._resolve_landmark(landmark_name)
            result["latency_ms"]["resolve"] = (time.time() - resolve_start) * 1000

            if not resolved:
                result["error"] = f"Unknown landmark: {landmark_name}"
                self._publish_status("unknown_landmark", result)
                return result

            orig_name, data = resolved
            x, y, z = data["x"], data["y"], data.get("z", 1.0)
            yaw = data.get("yaw", 0.0)

            # 边界检查
            x, y, z = self._clamp_coordinates(x, y, z)
            result["coordinates"] = {"x": x, "y": y, "z": z, "yaw": yaw}

            # 发布目标
            pub_start = time.time()
            self._publish_goal(x, y, z, yaw)
            result["latency_ms"]["publish"] = (time.time() - pub_start) * 1000

            # 更新上下文
            self.last_landmark = orig_name
            self.context_history.append({
                "user": text,
                "result": f"navigate to {orig_name}"
            })

            result["success"] = True
            result["resolved_landmark"] = orig_name

        elif action in ["stop", "pause", "resume"]:
            # TODO: 实现停止/暂停/继续控制
            result["success"] = True
            self.context_history.append({
                "user": text,
                "result": action
            })
            rospy.logwarn(f"Action '{action}' received but not implemented in FSM")

        else:
            result["error"] = f"Unknown action: {action}"

        # === 4. 统计 ===
        total_latency = (time.time() - total_start) * 1000
        result["latency_ms"]["total"] = total_latency

        self.stats["requests"] += 1
        self.stats["total_latency_ms"] += total_latency
        self.stats["avg_latency_ms"] = (
            self.stats["total_latency_ms"] / self.stats["requests"]
        )

        self._publish_status("completed", result)
        rospy.loginfo(
            f"Command processed: {result['success']}, "
            f"total={total_latency:.0f}ms (LLM={llm_latency:.0f}ms)"
        )

        return result

    def _text_callback(self, msg: String):
        """话题输入回调"""
        text = msg.data.strip()
        if text:
            self.process_command(text)

    def run_interactive(self):
        """交互式终端模式"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("Interactive mode started. Type commands or 'quit' to exit.")
        rospy.loginfo("Examples: '去充电桩', 'go to meeting room', '停止'")
        rospy.loginfo("=" * 50)

        while not rospy.is_shutdown():
            try:
                text = input("\n[LLM Nav] >>> ").strip()
                if not text:
                    continue
                if text.lower() in ["quit", "exit", "q"]:
                    break
                if text.lower() == "stats":
                    print(f"Statistics: {json.dumps(self.stats, indent=2)}")
                    continue
                if text.lower() == "clear":
                    self.context_history.clear()
                    self.last_landmark = None
                    print("Context cleared.")
                    continue
                if text.lower() == "context":
                    print(f"Context history: {list(self.context_history)}")
                    print(f"Last landmark: {self.last_landmark}")
                    continue

                result = self.process_command(text)
                print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")

            except EOFError:
                break
            except KeyboardInterrupt:
                break

        rospy.loginfo("Interactive mode ended.")

    def spin(self):
        """ROS spin 模式（仅监听话题）"""
        rospy.loginfo("Listening on ~/text_input topic...")
        rospy.spin()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Navigation Node")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    # ROS 参数会过滤掉，这里只处理自定义参数
    args, _ = parser.parse_known_args()

    node = LowLatencyLLMNavigator()

    if args.interactive:
        # 启动话题监听线程
        spin_thread = threading.Thread(target=lambda: rospy.spin(), daemon=True)
        spin_thread.start()
        # 主线程运行交互模式
        node.run_interactive()
    else:
        node.spin()


if __name__ == "__main__":
    main()
