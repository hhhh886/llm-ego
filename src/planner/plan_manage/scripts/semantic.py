#!/usr/bin/env python3
"""
Robot semantic landmark navigation (fixed).
Integrated with Ollama for natural language understanding.
"""
import json
import re
from typing import Dict

import requests


class SemanticNavigator:
    def __init__(self, ollama_url="http://127.0.0.1:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:1b-instruct-q4_0"

    def _clean_json_string(self, raw_string: str) -> str:
        """
        Cleaning function: extracts JSON object from Markdown or conversational text.
        """
        match = re.search(r"\{.*\}", raw_string, re.DOTALL)
        if match:
            return match.group(0)
        return raw_string

    def _validate_result(self, result: Dict) -> Dict:
        """
        验证并规范化 LLM 返回的结果。
        确保包含必需字段，值在合理范围内。
        """
        valid_actions = {"navigate", "stop", "pause", "unknown"}
        
        # 确保 action 字段存在且有效
        action = result.get("action", "unknown")
        if action not in valid_actions:
            action = "unknown"
        
        # 确保 landmark 字段存在
        landmark = result.get("landmark")
        if landmark is not None:
            landmark = str(landmark).strip()
            if not landmark:
                landmark = None
        
        # 确保 confidence 字段在 0-1 范围内
        try:
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0
        
        return {
            "action": action,
            "landmark": landmark,
            "confidence": confidence
        }

    def parse_navigation_intent(self, user_input: str) -> Dict:
        """Parse user navigation intent."""
        prompt = f"""你是一个机器人导航助手。
分析以下指令并返回JSON格式。
必须包含 keys: "action", "landmark", "confidence".
action 必须是: "navigate", "stop", "pause", "unknown" 之一.
confidence 是 0 到 1 之间的浮点数，表示解析的置信度。
landmark 是目标地点的名称（如果 action 是 navigate）。

用户指令：{user_input}

只返回JSON:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()["response"]

            cleaned_result = self._clean_json_string(result)
            parsed = json.loads(cleaned_result)
            
            # 验证并规范化结果
            return self._validate_result(parsed)

        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}")
            return {"action": "unknown", "landmark": None, "confidence": 0.0}
        except json.JSONDecodeError:
            print(f"JSON parse failed. Raw content: {result}")
            return {"action": "unknown", "landmark": None, "confidence": 0.0}
        except Exception as e:
            print(f"Parse intent failed: {e}")
            return {"action": "unknown", "landmark": None, "confidence": 0.0}

    def get_path_description(self, start: str, end: str) -> str:
        """Generate a short path description."""
        prompt = f"用一句话描述从{start}到{end}的路径。"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "max_tokens": 100},
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Path description failed: {e}")
            return f"无法生成从{start}到{end}的路径描述"


if __name__ == "__main__":
    nav = SemanticNavigator()
    test_commands = ["带我去充电桩", "停止移动", "去会议室开会"]
    print("=== Test: parse intent ===")
    for cmd in test_commands:
        result = nav.parse_navigation_intent(cmd)
        print(f"指令: {cmd}")
        print(f"解析: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 40)
    print("\n=== Test: path description ===")
    path_desc = nav.get_path_description("充电桩", "会议室")
    print(f"路径描述: {path_desc}")
