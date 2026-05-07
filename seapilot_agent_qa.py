
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
# 0. Metrics Infrastructure (Optimized)
# ==========================================
@dataclass
class AgentMetrics:
    # --- Cloud Metrics ---
    cloud_tokens: int = 0
    cloud_latency: float = 0.0
    
    # --- Edge Detail Metrics ---
    monitor_tokens: int = 0
    monitor_latency: float = 0.0   
    
    executor_tokens: int = 0
    executor_latency: float = 0.0  
    
    # --- Edge Summary Metrics ---
    edge_total_steps: int = 0
    edge_total_latency: float = 0.0 

    def __str__(self):
        system_overhead = self.edge_total_latency - (self.monitor_latency + self.executor_latency+ self.cloud_latency)
        if system_overhead < 0: system_overhead = 0.0
        
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
BASE_URL = os.getenv("PLANNER_BASE_URL", os.getenv("EDGE_TEXT_BASE_URL", "http://127.0.0.1:18085/v1"))
API_KEY = os.getenv("PLANNER_API_KEY", os.getenv("EDGE_TEXT_API_KEY", "EMPTY"))

CLOUD_MODEL_NAME = os.getenv("PLANNER_MODEL_NAME", "gpt-4-turbo-2024-04-09")
MONITOR_MODEL_PATH = os.getenv("MONITOR_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")
EXECUTOR_MODEL_PATH = os.getenv("EXECUTOR_MODEL_PATH", "Qwen2.5-3B-Instruct")
EDGE_TEXT_BASE_URL = os.getenv("EDGE_TEXT_BASE_URL", "http://127.0.0.1:18085/v1")
EDGE_TEXT_MODEL = os.getenv("EXECUTOR_MODEL_NAME", "Qwen3-4B-Instruct")
EDGE_VISION_BASE_URL = os.getenv("EDGE_VISION_BASE_URL", "http://127.0.0.1:18084/v1")
EDGE_VISION_MODEL = os.getenv("EDGE_VISION_MODEL", "Qwen3-VL-2B-Instruct")

# ==========================================
# 2. Prompts
# ==========================================

CLOUD_PLANNER_PROMPT = """
You are a High-Level Strategic Planner for Android automation.
Goal: {task_instruction}

Your job is to break this goal into a list of SEMANTIC MILESTONES (Navigation Steps).

{guidance}

For each milestone, you must define:
1. "instruction": The high-level navigation goal.
2. "expectation": The VISUAL STATE of the screen after this step.

**RULES:**
1. **Navigation Steps:** The early steps are for moving the UI (Open App -> Search -> Click).
2. **The Final Step:** The LAST milestone must be an instruction to **Identify/Read** the target information.
3. **NO COGNITIVE STEPS in Early Phase:** Do not use "Read" or "Verify" in early steps, only "Open", "Search", "Navigate".

**GOOD Examples:**
- Goal: "What quantity of quinoa flakes do I need for the recipe 'Spaghetti Carbonara' in the Joplin?"
  -> [
       {{"instruction": "Open Joplin app.", "expectation": "Joplin home screen is visible."}},
       {{"instruction": "Search for 'Spaghetti Carbonara' note.", "expectation": "Search results showing 'Spaghetti Carbonara' are visible."}}, 
       {{"instruction": "Open the 'Spaghetti Carbonara' note to view details.", "expectation": "The 'Spaghetti Carbonara' note is open and the relevant content is clearly visible on screen."}},
       {{"instruction": "Identify the quantity of quinoa flakes.", "expectation": "The specific text regarding quinoa flakes is visible."}} 
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

MONITOR_NAV_PROMPT = """
You are a strict Screen State Validator. 

Current Sub-goal: "{sub_goal}"
*** SUCCESS CRITERIA (Expected Visual State) ***: 
"{expectation}"

Context - Recent Action History:
{history}

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

MONITOR_QA_PROMPT = """
You are a Final Answer Extractor. The user is waiting for an answer.

=== GLOBAL USER QUESTION ===
"{global_goal}"
(Constraint: Answer format must match this request EXACTLY.)

=== CURRENT TASK ===
"{sub_goal}" (This is the final step intended to reveal the answer)

Current Screen Context:
{observation}

**CRITICAL DATA INTERPRETATION RULES**:
Apply these rules based on the app currently visible or mentioned in the task:
1. **OpenTracks**: The smallest time unit is **SECONDS**. Interpret "XX:YY" as **Minutes:Seconds** (e.g., "10:00" = 10 mins, NOT 10 hours).
2. **OpenTracks**: 1 mile = 1609.34 meters

**LOGIC FLOW**:
1. **CHECK FOR ANSWER**: 
   - Can you see the specific information requested in the "GLOBAL USER QUESTION" on this screen?
   - **Constraint**: If the question asks about content INSIDE a note/email, you must be viewing that content, not just a list title.
   
2. **DECIDE STATUS**:
   - **ANSWER_READY**: Yes, the text is clearly visible. -> Extract it.
   - **ONGOING**: No, the information is missing, obscured, or we are on the wrong page. -> Suggest scrolling or clicking to find it.

Output strictly in JSON:
{{
    "observation": "Brief screen description.",
    "status": "ANSWER_READY" | "ONGOING",
    "reasoning": "Why you can or cannot answer yet.",
    "answer_text": "THE EXACT ANSWER TEXT (or empty string if ONGOING).",
    "suggestion": "If ONGOING, how to make the answer visible?"
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
        start_time = time.time() 
        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0
            )
            latency = time.time() - start_time
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
            return [{"instruction": fallback_goal, "expectation": "Task Completed."}], 0, time.time() - start_time

# ==========================================
# 4. Visual Monitor (Updated for Dynamic Prompts)
# ==========================================
class VisualMonitor:
    def __init__(self, model_path: str):
        self.base_url = EDGE_VISION_BASE_URL
        self.model_name = EDGE_VISION_MODEL
        self.client = openai.OpenAI(api_key=os.getenv("EDGE_VISION_API_KEY", "EMPTY"), base_url=self.base_url)
        print(f"[System] Using Monitor API: {self.base_url} model={self.model_name}")

    def predict_status(self, template: str, global_goal: str, sub_goal: str, expectation: str, img: Image.Image, history: str) -> Tuple[Dict, int]:

        if "{global_goal}" in template:
            prompt = template.format(global_goal=global_goal, sub_goal=sub_goal, expectation=expectation, history=history, observation="Screenshots provided")
        else:
            prompt = template.format(sub_goal=sub_goal, expectation=expectation, history=history, observation="Screenshots provided")

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

            is_slider = "SeekBar" in cls_full or "Slider" in cls_full
            box_info = ""
            
            if is_slider:

                box = getattr(elem, 'bbox_pixels', None)
                if box:

                    box_info = f" Bbox:[{box.x_min},{box.y_min},{box.x_max},{box.y_max}]"

            info = f"[{i}] <{cls}>{box_info}" 
            if text: info += f" Text:\"{text}\""
            if desc: info += f" Desc:\"{desc}\""
            

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
        self.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print(f"[System] Using Planner API: {BASE_URL} model={CLOUD_MODEL_NAME}")
        self.rule_manager = RuleManager(
            edge_text_client=openai.OpenAI(api_key=API_KEY, base_url=EDGE_TEXT_BASE_URL),
            model_name=EDGE_TEXT_MODEL
        )
        
        self.planner = CloudPlanner(self.client)
        self.monitor = VisualMonitor(MONITOR_MODEL_PATH)
        self.executor = TextExecutor(EXECUTOR_MODEL_PATH)
        
        self.plan: List[Dict[str, str]] = [] 
        self.current_step_index = 0
        self.subgoal_step_count = 0
        self.max_steps_per_subgoal = 5 
        self.action_history: List[str] = [] 
        

        self.replan_count = 0
        self.max_replans = 1
        self.execution_trace: List[Dict] = []
        
        self.metrics = AgentMetrics()

    def step(self, goal: str) -> base_agent.AgentInteractionResult:

        step_start_time = time.time()  
        self.metrics.edge_total_steps += 1
        
        state = self.get_post_transition_state()
        ui_count = len(state.ui_elements)
        print(f"📱 [System] Current Screen UI Elements: {ui_count}")

        img = Image.fromarray(state.pixels)
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        max_dimension = 1024
        if max(img.size) > max_dimension:
            scale = max_dimension / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        if not self.plan:
            print(f"\n[Cloud] Logic Planning for goal: {goal}")
            planning_guidance = self.rule_manager.get_planning_guidance(goal, scope="qa")
            plan_data, p_tokens, p_latency = self.planner.plan(goal, guidance=planning_guidance)
            self.plan = plan_data
            self.metrics.cloud_tokens += p_tokens
            self.metrics.cloud_latency += p_latency
            print(f"[Cloud] Generated Plan: {json.dumps(self.plan, indent=2)}")
            self.current_step_index = 0
            self.execution_trace = [] 

        step_info = {
            "total_subgoals": len(self.plan),
            "current_subgoal_idx": self.current_step_index,
            "is_last_subgoal": (len(self.plan) > 0) and (self.current_step_index >= len(self.plan) - 1)
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
        
        if step_info["is_last_subgoal"]:
            print(f"🎯 [System] Phase: EXTRACTION (Final Step) -> Using QA Prompt")
            selected_template = MONITOR_QA_PROMPT
        else:
            print(f"🧭 [System] Phase: NAVIGATION -> Using Nav Prompt")
            selected_template = MONITOR_NAV_PROMPT

        monitor_res, m_tokens = self.monitor.predict_status(
            template=selected_template,
            global_goal=goal,
            sub_goal=current_sub_goal, 
            expectation=current_expectation, 
            img=img, 
            history=history_str
        )
        self.metrics.monitor_latency += (time.time() - t_monitor_start)
        self.metrics.monitor_tokens += m_tokens 
        
        status = monitor_res.get("status", "ONGOING").upper()
        
        current_trace_entry = {
            "step": self.metrics.edge_total_steps,
            "sub_goal": current_sub_goal,
            "sub_goal_idx": f"{self.current_step_index + 1}/{len(self.plan)}",
            "monitor": {
                "observation": monitor_res.get("observation", "No observation"),
                "status": status,
                "reasoning": monitor_res.get("reasoning", "")
            }
        }


        if status == "ANSWER_READY":
            if not step_info["is_last_subgoal"]:
                 print(f"🛑 [System] Rejected ANSWER_READY in non-final step. Forcing ONGOING.")
                 status = "ONGOING"
            else:
                answer_text = monitor_res.get("answer_text", "")
                if answer_text:
                    print(f"🗣️ [Monitor] Answer detected: {answer_text}")
                    answer_action = {"action_type": "answer", "text": answer_text}
                    current_trace_entry["action"] = f"Answered: {answer_text}"
                    self.execution_trace.append(current_trace_entry)
                    self.env.execute_action(json_action.JSONAction(**answer_action))
                    self.metrics.edge_total_latency += (time.time() - step_start_time)
                    print(self.metrics)
                    return base_agent.AgentInteractionResult(True, {**step_info, "info": "Task Completed via Monitor Answer"})


        if status == "FINISHED":
            print(f"✅ [Monitor] Sub-goal Finished.")
            self.current_step_index += 1
            self.subgoal_step_count = 0
            self.action_history = [] 
            self.execution_trace = []
            
            current_trace_entry["action"] = "No Action (Monitor Finished)"

            self.metrics.edge_total_latency += (time.time() - step_start_time)

            return base_agent.AgentInteractionResult(False, {**step_info, "info": "Sub-goal finished", "skip_cost": True})
            
        # ==========================================
        # Phase 2: Executor or Timeout/Replan
        # ==========================================
        
        if self.subgoal_step_count >= self.max_steps_per_subgoal:
            if self.replan_count < self.max_replans:
                print(f"⚠️ [System] Subgoal Timeout! Triggering Replan...")
                current_trace_entry["action"] = "No Action (Timeout Replan)"
                self.execution_trace.append(current_trace_entry)
                trace_str = self._format_trace_for_cloud()
                replan_guidance = self.rule_manager.get_planning_guidance(goal, scope="qa")
                new_plan, r_tokens, r_latency = self.planner.replan(goal, self.plan, trace_str, guidance=replan_guidance)
                self.metrics.cloud_tokens += r_tokens
                self.metrics.cloud_latency += r_latency
                self.plan = new_plan
                self.current_step_index = 0
                self.subgoal_step_count = 0
                self.execution_trace = []
                self.replan_count += 1
                self.metrics.edge_total_latency += (time.time() - step_start_time)
                return base_agent.AgentInteractionResult(False, {**step_info, "info": "Replanned", "skip_cost": True})
            else:

                print(f"❌ [System] Max Replans ({self.max_replans}) reached. Task Failed.")
                

                self.metrics.edge_total_latency += (time.time() - step_start_time)
                print(self.metrics)
                
                step_info["is_last_subgoal"] = True 
                
                return base_agent.AgentInteractionResult(True,{**step_info, "info": "Task Failed: Max Replans Reached", "success": False, "skip_cost": True})

        hint = monitor_res.get("suggestion", "")
        if hint: hint = f"Monitor says: {hint}. Reason: {monitor_res.get('reasoning','')}"
        

        print("\n" + "="*20 + " [Debug] UI Tree passed to Executor " + "="*20)

        debug_ui_str = self.executor._serialize_ui(state.ui_elements)
        print(debug_ui_str)
        print("="*60 + "\n")
  

        t_executor_start = time.time()
        action_res, e_tokens = self.executor.predict_action(current_sub_goal, state.ui_elements, hint)
        self.metrics.executor_latency += (time.time() - t_executor_start)
        self.metrics.executor_tokens += e_tokens
        
        final_action, action_desc = self._map_text_action(action_res, state.ui_elements)
        log_action = self._format_action_for_log(final_action, state.ui_elements)
        current_trace_entry["action"] = log_action
        self.execution_trace.append(current_trace_entry)
        
        if final_action:
            try:
                print(f"🚀 [Action] Executing: {log_action}")
                self.env.execute_action(json_action.JSONAction(**final_action))
                self.action_history.append(action_desc)
            except Exception as e:
                print(f"[Exec Error] {e}")
                self.action_history.append("Action Failed.")
        else:
            self.action_history.append("Action Failed: Invalid mapping.")

        self.subgoal_step_count += 1
        self.metrics.edge_total_latency += (time.time() - step_start_time)
        print(self.metrics)
        return base_agent.AgentInteractionResult(False, step_info)
    
    def _format_trace_for_cloud(self) -> str:
        lines = []
        for t in self.execution_trace:
            lines.append(f"--- Step {t['step']} ---")
            lines.append(f"=== [Goal {t['sub_goal_idx']}]: {t['sub_goal']} ===")
            lines.append(f"[Monitor Observation]: ```json\n{json.dumps(t['monitor'], indent=4)}\n```")
            lines.append(f"[Action] Executing: {t['action']}")
            lines.append("")
        return "\n".join(lines)

    def _format_action_for_log(self, action: Dict, ui_elements: List) -> Dict:
        if not action: return {"action_type": "None"}
        log_act = action.copy()
        idx = log_act.get("index")
        if isinstance(idx, int) and 0 <= idx < len(ui_elements):
            elem = ui_elements[idx]
            name = getattr(elem, 'text', "") or getattr(elem, 'content_description', "") or "Unknown Element"
            log_act["target"] = f"{name} (Index {idx})"
            del log_act["index"]
        return log_act

    def _map_text_action(self, raw: Dict, ui_elements: List) -> Tuple[Optional[Dict], str]:
        if raw and "action" in raw and isinstance(raw["action"], dict): raw = raw["action"]
        if raw and "answer" in raw and isinstance(raw["answer"], dict): raw = raw["answer"]
        if not raw or "action_type" not in raw: return None, "Invalid Action"
        action_type = raw.get("action_type")
        
        if action_type == "click":
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "click", "index": idx}, f"Clicked [{idx}] '{elem_text}'"
            return None, "Click failed: Invalid Index"

        if action_type == "input_text":
            text = raw.get("text", "")
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "input_text", "text": text, "index": idx}, f"Typed '{text}' into [{idx}]"
            return None, "Input failed: Invalid Index"

        if action_type == "scroll":
            d = raw.get("direction", "down").lower()
            return {"action_type": "scroll", "direction": d}, f"Scrolled {d}"
        
        if action_type == "open_app":
            app_name = raw.get("app_name", "Unknown App")
            return {"action_type": "open_app", "app_name": app_name}, f"Open App: {app_name}"

        if action_type == "keyboard_enter":
            return {"action_type": "keyboard_enter"}, "Pressed Enter"

        if action_type == "long_press":
            idx = raw.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ui_elements):
                elem_text = self._get_element_semantic_info(ui_elements[idx])
                return {"action_type": "long_press", "index": idx}, f"Long Pressed [{idx}] '{elem_text}'"
            return None, "Long Press failed: Invalid Index"

        if action_type == "navigate_home": return {"action_type": "navigate_home"}, "Pressed Home"
        if action_type == "navigate_back": return {"action_type": "navigate_back"}, "Pressed Back"

        return None, f"Unknown Action: {action_type}"

    def _get_element_semantic_info(self, element) -> str:
        try:
            return (getattr(element, 'text', "") or 
                    getattr(element, 'content_description', "") or 
                    (getattr(element, 'resource_id', "") or "").split('/')[-1])[:30].strip()
        except: return "Unknown"
