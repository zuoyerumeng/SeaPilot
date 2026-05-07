"""SeaPilot Agent for Android Operation Tasks."""
import os
import io
import base64
from PIL import Image
import json
import re
import math
import time
from typing import List, Dict, Optional, Any, Tuple
from absl import logging
import openai
from dataclasses import dataclass, field

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action
from android_world.agents.rules.rule_manager import RuleManager

# ==========================================
# 0. Metrics Infrastructure
# ==========================================
@dataclass
class AgentMetrics:
    cloud_tokens: int = 0
    cloud_latency: float = 0.0
    monitor_tokens: int = 0
    monitor_latency: float = 0.0
    executor_tokens: int = 0
    executor_latency: float = 0.0
    edge_total_steps: int = 0
    edge_total_latency: float = 0.0

    def __str__(self):
        system_overhead = self.edge_total_latency - (
            self.monitor_latency + self.executor_latency + self.cloud_latency
        )
        if system_overhead < 0:
            system_overhead = 0.0
        return (
            f"\n📊 --- Performance Report ---\n"
            f"   ☁️ [Cloud]    Tokens: {self.cloud_tokens:<6} | Latency: {self.cloud_latency:.2f}s\n"
            f"   -------------------------------------------\n"
            f"   👁️ [Monitor]  Tokens: {self.monitor_tokens:<6} | Time: {self.monitor_latency:.2f}s\n"
            f"   ✋ [Executor] Tokens: {self.executor_tokens:<6} | Time: {self.executor_latency:.2f}s\n"
            f"   ⚙️ [System]   Overhead: {system_overhead:.2f}s (Data Prep & Logic)\n"
            f"   -------------------------------------------\n"
            f"   🤖 [Total] All Steps: {self.edge_total_steps} | Latency: {self.edge_total_latency:.2f}s\n"
        )

# ==========================================
# 1. Configuration Area
# ==========================================
PLANNER_BASE_URL = os.environ.get("PLANNER_BASE_URL", "https://4.0.wokaai.com/v1")
PLANNER_API_KEY = os.environ.get("PLANNER_API_KEY", "EMPTY")
CLOUD_MODEL_NAME = os.environ.get("PLANNER_MODEL_NAME", "gpt-4-turbo-2024-04-09")
MONITOR_MODEL_PATH = os.environ.get("MONITOR_VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
EXECUTOR_MODEL_PATH = os.environ.get("EXECUTOR_MODEL_NAME", "Qwen3-4B-Instruct")
EDGE_TEXT_BASE_URL = os.environ.get("EDGE_TEXT_BASE_URL", "http://127.0.0.1:18085/v1")
EDGE_TEXT_MODEL = os.environ.get("EXECUTOR_MODEL_NAME", "Qwen3-4B-Instruct")
EDGE_VISION_BASE_URL = os.environ.get("EDGE_VISION_BASE_URL", "http://127.0.0.1:18084/v1")
EDGE_VISION_MODEL = os.environ.get("EDGE_VISION_MODEL", "Qwen3-VL-2B-Instruct")

# ==========================================
# 2. Prompts
# ==========================================
CLOUD_PLANNER_PROMPT = """
You are a High-Level Strategic Planner for Android automation.
Goal: {task_instruction}

Your job is to break this goal into a list of SEMANTIC MILESTONES (Action Steps).

{guidance}

For each milestone, you must define:
1. "instruction": The high-level action goal.
2. "expectation": The VISUAL STATE of the screen after this step.

**RULES:**
1. **Navigation Steps:** The early steps are for moving the UI (Open App -> Navigate).
2. **The Final Step:** The LAST milestone must perform the required action (delete/edit/create/etc.).
3. **CONCISE:** Keep the plan concise. Avoid redundant steps.

**GOOD Examples:**
- Goal: "In Simple Calendar Pro, delete the calendar event on 2023-10-24 with title 'Meeting'."
  -> [
       {{"instruction": "Open Simple Calendar Pro app.", "expectation": "Simple Calendar Pro home screen is visible."}},
       {{"instruction": "Navigate to 2023-10-24 and find the 'Meeting' event.", "expectation": "The 'Meeting' event is visible on the calendar."}},
       {{"instruction": "Delete the 'Meeting' event.", "expectation": "The event is deleted and no longer visible."}}
     ]

Output strictly in JSON format:
[
  {{"instruction": "...", "expectation": "..."}},
  ...
]
"""

REPLAN_PROMPT_TEMPLATE = """
You are a High-Level Strategic Planner for Android automation.
The previous plan FAILED to execute correctly. You need to analyze the failure trace and decide whether the failure is caused by an Environment Assumption Gap (EAG).

**GOAL**: {goal}

{guidance}

=== FAILED PLAN ===
{old_plan}

=== EXECUTION TRACE (History of what happened) ===
{trace}

**TASK:**
1. **DIAGNOSE**: Read the [Monitor Observation] and [Action] in the trace.
    - Determine whether the root cause is an **EAG** (planner-side wrong assumption about UI/app behavior) or an execution-side transient issue.
2. **BRANCH**:
    - **If NOT EAG**: Do not modify the plan. Return a retry directive for edge-side re-execution from current screen.
    - **If EAG**: Reflect on the wrong assumption, induce a structured environment rule, and generate a revised plan.
3. **RULE INDUCTION (EAG only)**:
    - Scope: which app/task condition this rule applies to.
    - Constraint: the actual environment behavior that invalidated the old assumption.
    - Adaptation: concrete planning strategy change to satisfy the constraint.

**OUTPUT FORMAT:**
Your response MUST strictly adhere to the following structure:

Case A (Not EAG):
Diagnosis: EXECUTION_ERROR
Action: RETRY
Reason: <one-sentence reason>

Case B (EAG):
Diagnosis: EAG
Wrong Assumption: <what the planner incorrectly assumed>
Rule.Scope: <applicable task/app scope>
Rule.Constraint: <actual environment behavior>
Rule.Adaptation: <planning adjustment>
Reflection: <brief summary>
Plan: <JSON list of milestones>
"""

MONITOR_NAV_BASE_PROMPT = """
You are a strict Screen State Validator for Android operation tasks.

Current Sub-goal: "{sub_goal}"
*** SUCCESS CRITERIA (Expected Visual State) ***:
"{expectation}"

Context - Recent Action History:
{history}

{dynamic_rules}

Task:
1. **OBSERVE**: Look at the screenshot. Does it match the "SUCCESS CRITERIA"?
2. **VERIFY HISTORY**: Did we actually perform an action?
3. **JUDGE**:
   - Matches Expectation AND History confirms action -> FINISHED.
   - Starting point / Mismatch -> ONGOING.

** Capabilities Note **:
The Executor can OPEN APP, CLICK, TYPE, SCROLL, LONG PRESS, and use SYSTEM KEYS (Back/Home).
1. **"OPEN APP" is a MAGIC SPELL**:
   - The Executor can launch any app directly by name from ANYWHERE.
   - IF goal is "Open X app" AND app is not visible -> SUGGEST "open_app" command. DO NOT SUGGEST SCROLLING.
2. **"HOME SCREEN NAV" (Special Context)**:
   - When on the **Home Screen**:
   - **SCROLL DOWN** -> Opens **App Drawer**.
   - **SCROLL UP** -> Opens **Quick Settings**.

Output strictly in JSON format:
{{
    "observation": "Brief description of current screen.",
    "status": "FINISHED" | "ONGOING",
    "reasoning": "Detailed analysis of why the screen matches or fails the expectation.",
    "suggestion": "If ONGOING, what is the exact next move?"
}}
"""

EXECUTOR_PROMPT_TEMPLATE = """
You are a precise UI Operator.
Your task is to execute the next action based on the provided UI Elements List and the Analysis from the Monitor Agent.

Current Sub-goal: "{sub_goal}"

==================================================
*** ANALYSIS FROM MONITOR AGENT (Read Carefully) ***
{hint_text}
==================================================
*** SCREEN UI ELEMENTS (Index | Type | Text | Description) ***
{ui_tree}
==================================================

Task:
1. **READ**: Read the Monitor's analysis above to understand the target element.
2. **SEARCH**: Scan the "SCREEN UI ELEMENTS" list above. Find the element (by Text or Description) that matches the Monitor's hint.
3. **ACT**: Select the best action using the matching element's INDEX.

Available Actions:
[Interaction]
1. {{"action_type": "click", "index": <index>}}
   - Tap a UI element.
2. {{"action_type": "input_text", "text": "...", "index": <index>}}
   - Type text into a field (handles focus automatically).
3. {{"action_type": "long_press", "index": <index>}}
   - Long press an element (use only when necessary).

[Navigation]
4. {{"action_type": "open_app", "app_name": "..."}}
   - Directly launch an app by name.
5. {{"action_type": "scroll", "direction": "<direction>"}}
   - "down": Scroll down / Open App Drawer.
   - "up": Scroll up / Open Quick Settings.
6. {{"action_type": "navigate_back"}}
   - System Back button. Use this to close keyboards, popups, or go back.
7. {{"action_type": "navigate_home"}}
   - Go to Home Screen.

Output strictly in JSON format.
"""

# Static dynamic rules for specific task types (injected into MONITOR_NAV_BASE_PROMPT)
RULE_BRIGHTNESS = ""
RULE_DELETE = """
[Delete Rule]: To delete items, look for a Delete button, trash icon, or try Long Press to reveal delete options.
"""
RULE_RESEND = ""

# ==========================================
# 3. Cloud Planner
# ==========================================
class CloudPlanner:
    def __init__(self, client):
        self.client = client
        self.model = CLOUD_MODEL_NAME

    def plan(self, task_instruction: str, guidance: str = "") -> Tuple[List[Dict[str, str]], int, float]:
        print(f"[Cloud] Generating Logic Plan...")
        prompt_text = CLOUD_PLANNER_PROMPT.format(task_instruction=task_instruction, guidance=guidance)
        return self._call_llm(prompt_text, task_instruction)

    def replan(self, goal: str, old_plan: List[Dict], trace: str, guidance: str = "") -> Tuple[List[Dict[str, str]], int, float]:
        print(f"[Cloud] ⚠️ Analyzing Failure & Re-planning...")
        prompt_text = REPLAN_PROMPT_TEMPLATE.format(
            goal=goal,
            guidance=guidance,
            old_plan=json.dumps(old_plan, indent=2),
            trace=trace
        )
        return self._call_llm(prompt_text, goal)

    def _call_llm(self, prompt: str, fallback_goal: str) -> Tuple[List[Dict[str, str]], int, float]:
        import time as _time
        start_time = _time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            latency = _time.time() - start_time
            tokens = response.usage.total_tokens if response.usage else 0

            content = response.choices[0].message.content.strip()
            if "Reflection:" in content:
                print(f"🤔 [Cloud Reflection]: {content.split('Plan:')[0].replace('Reflection:', '').strip()}")

            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group()), tokens, latency

            print("⚠️ [Cloud] JSON parsing failed, applying fallback recovery.")
            return [{"instruction": "Navigate Home", "expectation": "Home screen visible"}], tokens, latency
        except Exception as e:
            print(f"[Cloud Error] {e}")
            return [{"instruction": fallback_goal, "expectation": "Task Completed."}], 0, _time.time() - start_time

# ==========================================
# 4. Visual Monitor (Same)
# ==========================================
class VisualMonitor:
    def __init__(self, model_path: str):
        self.base_url = EDGE_VISION_BASE_URL
        self.model_name = EDGE_VISION_MODEL
        self.client = openai.OpenAI(api_key=os.getenv("EDGE_VISION_API_KEY", "EMPTY"), base_url=self.base_url)
        print(f"[System] Using Monitor API: {self.base_url} model={self.model_name}")

# 🔥 [修改] 增加 template 参数
    def predict_status(self, template: str, sub_goal: str, expectation: str, img: Image.Image, history: str) -> Tuple[Dict, int]:
        prompt = template.format(sub_goal=sub_goal, expectation=expectation, history=history)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=0.0,
        )
        output = (response.choices[0].message.content or "").strip()
        print(f"[Monitor Observation]: {output}")

        tokens = response.usage.total_tokens if response.usage else 0
        return self._extract_json(output), tokens
    def _extract_json(self, text: str) -> Dict:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group())
            return {}
        except: return {}

# ==========================================
# 5. Text Executor (Same)
# ==========================================
class TextExecutor:
    def __init__(self, model_path: str):
        self.base_url = EDGE_TEXT_BASE_URL
        self.model_name = EDGE_TEXT_MODEL
        self.client = openai.OpenAI(api_key=os.getenv("EDGE_TEXT_API_KEY", "EMPTY"), base_url=self.base_url)
        print(f"[System] Using Executor API: {self.base_url} model={self.model_name}")

    def predict_action(self, sub_goal: str, ui_elements: List, hint: str) -> Tuple[Dict, int]:
        ui_tree_str = self._serialize_ui(ui_elements)
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(sub_goal=sub_goal, hint_text=hint, ui_tree=ui_tree_str)
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful Android UI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                output = (response.choices[0].message.content or "").strip()
                print(f"[Executor Thought]: {output}")
                tokens = response.usage.total_tokens if response.usage else 0
                return self._extract_json(output), tokens
            except Exception as e:
                last_error = e
                print(f"[Executor Error] attempt {attempt + 1}/3: {e}")
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))

        print(f"[Executor Error] all retries failed: {last_error}")
        return {}, 0

    def _serialize_ui(self, ui_elements: List) -> str:
        lines = []
        for i, elem in enumerate(ui_elements):
            if hasattr(elem, 'is_visible') and not elem.is_visible: continue
            
            text = getattr(elem, 'text', "") or ""
            desc = getattr(elem, 'content_description', "") or ""
            cls_full = getattr(elem, 'class_name', "View")
            cls = cls_full.split('.')[-1]
            
            # 🔥 [新增] 针对滑动条提取 Bbox 坐标
            # 只要类名里带 SeekBar 或者 Slider，就认为是需要物理操作的控件
            is_slider = "SeekBar" in cls_full or "Slider" in cls_full
            box_info = ""
            
            if is_slider:
                # 获取 AndroidWorld 元素的真实像素边界
                box = getattr(elem, 'bbox_pixels', None)
                if box:
                    # 格式: [x_min, y_min, x_max, y_max]
                    box_info = f" Bbox:[{box.x_min},{box.y_min},{box.x_max},{box.y_max}]"

            info = f"[{i}] <{cls}>{box_info}" 
            if text: info += f" Text:\"{text}\""
            if desc: info += f" Desc:\"{desc}\""
            
            # 过滤逻辑：保留有内容的，或者虽然没内容但是是 Slider 的
            if text or desc or is_slider:
                lines.append(info)
        
        if len(lines) > 200:
            return "\n".join(lines[:60] + ["..."] + lines[-20:])
        return "\n".join(lines)

    def _extract_json(self, text: str) -> Dict:
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```', '', text)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group())
            return json.loads(text)
        except: return {}

# ==========================================
# 6. Main SeaPilot Agent (With Replan)
# ==========================================
class SeaPilotAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv, name: str = 'SeaPilotAgent'):
        super().__init__(env, name)
        self.client = openai.OpenAI(api_key=PLANNER_API_KEY, base_url=PLANNER_BASE_URL)
        print(f"[System] Using Planner API: {PLANNER_BASE_URL} model={CLOUD_MODEL_NAME}")
        
        self.planner = CloudPlanner(self.client)
        self.monitor = VisualMonitor(MONITOR_MODEL_PATH)
        self.executor = TextExecutor(EXECUTOR_MODEL_PATH)
        self.rule_manager = RuleManager(
            edge_text_client=openai.OpenAI(
                api_key=os.getenv("EDGE_TEXT_API_KEY", "EMPTY"),
                base_url=EDGE_TEXT_BASE_URL,
            ),
            model_name=EDGE_TEXT_MODEL,
        )
        self.active_rule_ids: List[str] = []
        
        self.plan: List[Dict[str, str]] = [] 
        self.current_step_index = 0
        self.subgoal_step_count = 0
        self.max_steps_per_subgoal = 5 
        self.action_history: List[str] = [] 
        
        # 🔥 [新增] Replan 相关状态
        self.replan_count = 0
        self.max_replans = 1
        self.execution_trace: List[Dict] = [] # 存储详细执行轨迹供反思
        
        self.metrics = AgentMetrics()

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        # [A] 整个步骤开始计时
        step_start_time = time.time()  
        self.metrics.edge_total_steps += 1
        
        # --- 系统开销部分：环境感知 ---
        state = self.get_post_transition_state()

        img = Image.fromarray(state.pixels)
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")

        # 强制压缩图片分辨率
        max_dimension = 1024
        if max(img.size) > max_dimension:
            scale = max_dimension / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # --- Cloud Planning (仅在首次触发) ---
        if not self.plan:
            print(f"\n[Cloud] Logic Planning for goal: {goal}")
            planning_guidance = self.rule_manager.get_planning_guidance(goal, scope="op")
            self.active_rule_ids = list(self.rule_manager.last_selected_ids)
            merged_guidance = planning_guidance  # static GUIDANCE excluded; controlled entirely by rules.json via RULE_SCOPE_ROUND
            plan_data, p_tokens, p_latency = self.planner.plan(goal, guidance=merged_guidance)
            self.plan = plan_data
            self.metrics.cloud_tokens += p_tokens
            self.metrics.cloud_latency += p_latency
            print(f"[Cloud] Generated Plan: {json.dumps(self.plan, indent=2)}")
            self.current_step_index = 0
            self.execution_trace = [] # 初始化 Trace

        # 准备 step_info
        total_subgoals = len(self.plan) if self.plan else 0
        current_subgoal_idx = self.current_step_index 
        
        step_info = {
            "total_subgoals": total_subgoals,
            "current_subgoal_idx": current_subgoal_idx,
            "is_last_subgoal": (total_subgoals > 0) and (current_subgoal_idx >= total_subgoals - 1)
        }

        if self.current_step_index >= len(self.plan):
            return base_agent.AgentInteractionResult(True, {**step_info, 'summary': 'All milestones completed.'})

        current_step_data = self.plan[self.current_step_index]
        current_sub_goal = current_step_data.get("instruction", "Unknown Task")
        current_expectation = current_step_data.get("expectation", "Task completed")

        print(f"\n=== [Goal {self.current_step_index+1}/{len(self.plan)}]: {current_sub_goal} ===")
        
        # ==========================================
        # Phase 1: Visual Monitor (Eyes)
        # ==========================================
        history_str = "\n".join([f"- {h}" for h in self.action_history[-3:]]) if self.action_history else "None (Start of Task)"
        
        t_monitor_start = time.time()

        # 🔥 [Dynamic Prompt Construction] 核心修改区
        # 1. 初始化空规则
        dynamic_rules = ""
        goal_lower = goal.lower() # 这里的 goal 是 step 函数入参的全局目标

        # 2. 根据全局目标关键词注入规则
        if "brightness" in goal_lower:
            dynamic_rules += RULE_BRIGHTNESS
        
        if "delete" in goal_lower:
            dynamic_rules += RULE_DELETE
        
        if "resend" in goal_lower:
            dynamic_rules += RULE_RESEND

        # if "markor" in goal_lower:
        #     dynamic_rules += RULE_MARKOR_CREATE

        monitor_rule_block = self.rule_manager.get_monitoring_rules(goal, max_rules=3)
        if monitor_rule_block:
            dynamic_rules += "\n" + monitor_rule_block

        # 3. 替换 Base Prompt 中的占位符
        # 注意：这里使用 replace 而不是 format，是为了避免与 Prompt 中其他的 {} 冲突，
        # 且我们只替换 dynamic_rules，剩下的 sub_goal 等留给 monitor 内部去 format
        selected_template = MONITOR_NAV_BASE_PROMPT.replace("{dynamic_rules}", dynamic_rules)

        # 4. 调用 Monitor，传入构建好的 Template
        monitor_res, m_tokens = self.monitor.predict_status(
            template=selected_template,   # 🔥 传入新的 template
            sub_goal=current_sub_goal, 
            expectation=current_expectation, 
            img=img, 
            history=history_str
        )
        self.metrics.monitor_latency += (time.time() - t_monitor_start)
        self.metrics.monitor_tokens += m_tokens 
        
        status = monitor_res.get("status", "ONGOING").upper()

        if status == "FINISHED" and not self.action_history:
            print(f"⚠️ [Monitor] Forced ONGOING: The FINAL subgoal cannot finish without action.")
            status = "ONGOING"

        # 🔥 [新增] 记录 Monitor 思考（无论是否 Finished）
        current_trace_entry = {
            "step": self.metrics.edge_total_steps,
            "sub_goal": current_sub_goal,
            "sub_goal_idx": f"{self.current_step_index + 1}/{len(self.plan)}",
            "monitor": {
                "observation": monitor_res.get("observation", "No observation"),
                "status": status
            }
        }

        if status == "FINISHED":
            print(f"✅ [Monitor] Sub-goal Finished.")
            self.current_step_index += 1
            self.subgoal_step_count = 0
            self.action_history = [] 
            
            # 更新 Info
            step_info["current_subgoal_idx"] = self.current_step_index
            step_info["is_last_subgoal"] = (total_subgoals > 0) and (self.current_step_index >= total_subgoals - 1)
            
            # 将 Finished 状态也记录进 Trace
            current_trace_entry["action"] = "No Action (Monitor Finished)"
            self.execution_trace.append(current_trace_entry)

            self.metrics.edge_total_latency += (time.time() - step_start_time)
            print(self.metrics) 
            return base_agent.AgentInteractionResult(False, {**step_info, "info": "Sub-goal finished", "skip_cost": True})
# =================================================================
        # 🔥【关键修复】：在执行动作前检查超时
        # =================================================================
        # 此时 Step 5 刚开始，subgoal_step_count 为 4。
        # 如果 max=4，则 4 >= 4 成立，拦截 Executor，直接 Replan。
        if self.subgoal_step_count >= self.max_steps_per_subgoal:
            if self.replan_count < self.max_replans:
                print(f"⚠️ [System] Subgoal Timeout ({self.subgoal_step_count}/{self.max_steps_per_subgoal})! Triggering Replan...")

                # 1. 记录当前 Monitor 的观察结果到 Trace，这对 Replan 至关重要
                # 因为没有执行动作，我们标记 Action 为 Timeout
                current_trace_entry["action"] = "No Action (Timeout Replan)"
                self.execution_trace.append(current_trace_entry)

                # 2. 格式化 Trace
                trace_str = self._format_trace_for_cloud()
                replan_guidance = self.rule_manager.get_planning_guidance(goal, scope="op")
                self.active_rule_ids = list(self.rule_manager.last_selected_ids)
                merged_guidance = replan_guidance  # static GUIDANCE excluded; controlled entirely by rules.json via RULE_SCOPE_ROUND
                
                # 3. 调用 Replan
                new_plan, r_tokens, r_latency = self.planner.replan(
                    goal, 
                    self.plan, 
                    trace_str,
                    guidance=merged_guidance
                )
                
                # 4. 更新 Metrics & State
                self.metrics.cloud_tokens += r_tokens
                self.metrics.cloud_latency += r_latency
                print(f"[Cloud] New Plan Applied: {json.dumps(new_plan, indent=2)}")
                
                self.plan = new_plan
                self.current_step_index = 0
                self.subgoal_step_count = 0
                self.execution_trace = []
                self.replan_count += 1
                
                # 更新 info
                step_info["total_subgoals"] = len(self.plan)
                step_info["current_subgoal_idx"] = 0
                step_info["is_last_subgoal"] = (len(self.plan) > 0) and (0 >= len(self.plan) - 1)

                # 🔥🔥🔥【新增修复】：在 return 前累加这一步消耗的总时间 🔥🔥🔥
                # 这一步虽然没有执行 Action，但消耗了 Monitor 和 Cloud Replan 的时间，必须计入 Total Latency
                self.metrics.edge_total_latency += (time.time() - step_start_time)
                print(self.metrics) # 打印最新的 Metrics 方便调试
                
                # 返回 skip_cost=True，这一步只算作“思考”，不计入 Runner 步数
                return base_agent.AgentInteractionResult(False, {**step_info, "info": "Replanned", "skip_cost": True})

            else:
                # Replan 次数用尽 -> 任务失败
                print(f"❌ [System] Max Replans ({self.max_replans}) reached. Task Failed.")
                if self.active_rule_ids:
                    self.rule_manager.update_stats(self.active_rule_ids, success=False)
                
                # 补齐时间统计 (虽然失败了，但时间还是要算的)
                self.metrics.edge_total_latency += (time.time() - step_start_time)
                print(self.metrics)
                
                # 更新 info，标记为失败
                # 我们不需要改 index，直接 done 即可
                step_info["is_last_subgoal"] = True # 逻辑上结束了
                
                # 🔥 返回 done=True，结束任务
                return base_agent.AgentInteractionResult(True,{**step_info, "info": "Task Failed: Max Replans Reached", "success": False, "skip_cost": True})
            
        # ==========================================
        # Phase 2: Text Executor (Hands)
        # ==========================================
        hint = monitor_res.get("suggestion", "")
        if hint: hint = f"Monitor says: {hint}. Reason: {monitor_res.get('reasoning','')}"
        
        # 🔥🔥🔥 【插入这段代码】 🔥🔥🔥
        print("\n" + "="*20 + " [Debug] UI Tree passed to Executor " + "="*20)
        # 临时调用一次 _serialize_ui 看看实际传给大模型的是什么
        debug_ui_str = self.executor._serialize_ui(state.ui_elements)
        print(debug_ui_str)
        print("="*60 + "\n")
        # 🔥🔥🔥 【插入结束】 🔥🔥🔥

        t_executor_start = time.time()
        action_res, e_tokens = self.executor.predict_action(current_sub_goal, state.ui_elements, hint)
        self.metrics.executor_latency += (time.time() - t_executor_start)
        self.metrics.executor_tokens += e_tokens
        
        final_action, action_desc = self._map_text_action(action_res, state.ui_elements)

        # 🔥 [新增] 完善 Trace 记录 (把 index 换成 element text)
        log_action = self._format_action_for_log(final_action, state.ui_elements)
        current_trace_entry["action"] = log_action
        self.execution_trace.append(current_trace_entry)
        
        if final_action:
            try:
                print(f"🚀 [Action] Executing: {log_action}")
                
                # 🔥 [最终修正版：Subprocess 方案]
                if final_action.get("action_type") == "swipe":
                    # 1. 获取 raw_command (例如 "input swipe 100 200 ...")
                    # 注意：根据你的代码逻辑，确认这里是用 'command' 还是 'raw_command'
                    # 之前你的代码存的是 'command'
                    inner_cmd = final_action.get("command")
                    
                    if inner_cmd:
                        # 2. 拼接完整的 adb 指令
                        # 如果是模拟器/真机，通常不需要 -s <serial>，除非连了多台设备
                        # ✅ 直接使用刚刚找到的路径
                        adb_bin = os.getenv("ADB_PATH", "adb")

                        full_cmd = f"{adb_bin} shell {inner_cmd}"
                        
                        print(f"   [System] 🚀 Physical Swipe via Subprocess: {full_cmd}")
                        
                        try:
                            # 3. 调用系统命令行执行
                            # shell=True 允许直接传字符串，虽然不安全但在 benchmark 里最方便
                            subprocess.run(full_cmd, shell=True, check=True, timeout=5)
                        except subprocess.CalledProcessError as e:
                            print(f"   [Error] ADB command failed: {e}")
                        except Exception as e:
                            print(f"   [Error] Unexpected error: {e}")
                    else:
                        print("   [Error] Swipe action missing 'command' field!")

                else:
                    # 普通动作 (Click, Type, Scroll) 依然走官方通道
                    self.env.execute_action(json_action.JSONAction(**final_action))
                self.action_history.append(action_desc)     
            except Exception as e:
                print(f"[Exec Error] {e}")
                self.action_history.append("Action Failed.")
        else:
            self.action_history.append("Action Failed: Invalid mapping.")

        self.subgoal_step_count += 1
        
        # [A] 整个步骤计时结束
        step_duration = time.time() - step_start_time
        self.metrics.edge_total_latency += step_duration
        
        print(self.metrics)
        return base_agent.AgentInteractionResult(False, step_info)
    
    # 🔥 [新增] 辅助函数：把 Trace 列表转成云端易读的字符串
    def _format_trace_for_cloud(self) -> str:
        lines = []
        for t in self.execution_trace:
            lines.append(f"--- Step {t['step']} ---")
            lines.append(f"=== [Goal {t['sub_goal_idx']}]: {t['sub_goal']} ===")
            lines.append(f"[Monitor Observation]: ```json\n{json.dumps(t['monitor'], indent=4)}\n```")
            lines.append(f"[Action] Executing: {t['action']}")
            lines.append("")
        return "\n".join(lines)

    # 🔥 [新增] 辅助函数：把 Action index 换成 text
    def _format_action_for_log(self, action: Dict, ui_elements: List) -> Dict:
        if not action: return {"action_type": "None"}
        
        log_act = action.copy()
        idx = log_act.get("index")
        
        if isinstance(idx, int) and 0 <= idx < len(ui_elements):
            # 获取元素名称
            elem = ui_elements[idx]
            name = getattr(elem, 'text', "") or getattr(elem, 'content_description', "") or "Unknown Element"
            # 替换 index 为 target
            log_act["target"] = f"{name} (Index {idx})"
            del log_act["index"]
        
        return log_act

    def _map_text_action(self, raw: Dict, ui_elements: List) -> Tuple[Optional[Dict], str]:
        # [新增] 1. 自动解包
        if raw and "action" in raw and isinstance(raw["action"], dict):
            raw = raw["action"]
        if raw and "answer" in raw and isinstance(raw["answer"], dict):
            raw = raw["answer"]
            
        if not raw or "action_type" not in raw: return None, "Invalid Action"
        action_type = raw.get("action_type")
        
        # 1. Handle Click
        if action_type == "click":
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "click", "index": idx}, f"Clicked [{idx}] '{elem_text}'"
            return None, "Click failed: Invalid Index"

        # 2. Handle Input Text
        if action_type == "input_text":
            text = raw.get("text", "")
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "input_text", "text": text, "index": idx}, f"Typed '{text}' into [{idx}]"
            return None, "Input failed: Invalid Index"

        # 3. Handle Scroll
        if action_type == "scroll":
            d = raw.get("direction", "down").lower()
            return {"action_type": "scroll", "direction": d}, f"Scrolled {d}"
        
        
        # 4. Handle Open App
        if action_type == "open_app":
            app_name = raw.get("app_name", "Unknown App")
            return {"action_type": "open_app", "app_name": app_name}, f"Open App: {app_name}"

        # 6. Handle Long Press
        if action_type == "long_press":
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "long_press", "index": idx}, f"Long Pressed [{idx}] '{elem_text}'"
            return None, "Long Press failed: Invalid Index"

        # 7. Handle System Keys
        if action_type == "navigate_home": return {"action_type": "navigate_home"}, "Pressed Home"
        if action_type == "navigate_back": return {"action_type": "navigate_back"}, "Pressed Back"

        # 🔥 [新增] 处理 Swipe 动作逻辑
        if action_type == "swipe":
            idx = raw.get("index")
            direction = raw.get("direction", "right").lower()
            
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem = ui_elements[idx]
                # 获取刚才在 _serialize_ui 里不仅提取了，而且存在 elem 对象里的 bbox
                box = getattr(elem, 'bbox_pixels', None)
                
                if box:
                    # 1. 计算中心点和宽高
                    center_x = int((box.x_min + box.x_max) / 2)
                    center_y = int((box.y_min + box.y_max) / 2)
                    width = box.x_max - box.x_min
                    height = box.y_max - box.y_min
                    
                    # 2. 定义安全内缩 (Padding) - 防止滑出界
                    pad_x = int(width * 0.05)  # 左右留 5%
                    pad_y = int(height * 0.05) # 上下留 5%
                    
                    # 3. 初始化起点和终点 (默认是中心不动)
                    start_x, start_y = center_x, center_y
                    end_x, end_y = center_x, center_y
                    
                    # 4. 根据方向计算轨迹
                    if direction == "right":  # 往右滑 (值变大)
                        start_x = box.x_min + pad_x
                        end_x   = box.x_max
                    elif direction == "left": # 往左滑 (值变小)
                        start_x = box.x_max - pad_x
                        end_x   = box.x_min
                    elif direction == "down": # 往下滑 (例如下拉刷新)
                        start_y = box.y_min + pad_y
                        end_y   = box.y_max
                    elif direction == "up":   # 往上滑 (例如上滑解锁)
                        start_y = box.y_max - pad_y
                        end_y   = box.y_min

                    # 5. 生成 ADB 指令
                    # 持续 400ms 是比较像人类的手势
                    cmd = f"input swipe {start_x} {start_y} {end_x} {end_y} 400"
 
                    return {
                        "action_type": "swipe", # 特殊标记
                        "command": cmd
                    }, f"Swiping {direction} on SeekBar [{idx}]"
                else:
                    return None, "Swipe failed: Element has no Bbox"
            return None, "Swipe failed: Invalid Index"
        
        return None, f"Unknown Action: {action_type}"

    def _get_element_semantic_info(self, element) -> str:
        try:
            return (getattr(element, 'text', "") or 
                    getattr(element, 'content_description', "") or 
                    (getattr(element, 'resource_id', "") or "").split('/')[-1])[:30].strip()
        except: return "Unknown"