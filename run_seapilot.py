"""Runner for SeaPilot Agent (Dynamic Routing)."""

import os
import sys
import time
import random
import numpy as np
from absl import app
from absl import flags
from absl import logging as absl_logging
from android_world.env import env_launcher
from android_world import registry



FLAGS = flags.FLAGS

flags.DEFINE_string('adb_path', os.path.expanduser(''), 'Path to ADB')
flags.DEFINE_string('task', 'ContactsAddContact', 'Task name')
flags.DEFINE_integer('console_port', 5554, 'Emulator port')
flags.DEFINE_integer('seed', 30, 'Random seed for reproducibility')


class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

def main(_):
    absl_logging.set_verbosity(absl_logging.ERROR)
    if FLAGS.adb_path:
        os.environ["ADB_PATH"] = FLAGS.adb_path
    
    results_dir = "results"
    log_file_path = os.path.join(results_dir, f"{FLAGS.task}.txt") 
    sys.stdout = DualLogger(log_file_path)

    print(f"🚀 Starting Task Runner for: {FLAGS.task}")
    print(f"📄 Logging to: {log_file_path}")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    print("Initializing Android Environment...")
    env = env_launcher.load_and_setup_env(
        console_port=FLAGS.console_port,
        emulator_setup=False, 
        adb_path=FLAGS.adb_path
    )
    env.reset(go_home=True)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    task_cls = aw_registry.get(FLAGS.task)
    if not task_cls:
        raise ValueError(f"Task {FLAGS.task} not found.")
    
    params = task_cls.generate_random_params()
    task = task_cls(params)
    task.initialize_task(env)
    
    final_goal_text = task.goal
    print(f"🎯 Goal: {final_goal_text}")

    
    print("\n⚡ [Router] Analyzing Task Type based on Goal Syntax...")
    
    if "?" in final_goal_text:
        print("💡 Detected '?' in goal -> Routing to **QA Agent** (seapilot_agent_qa)")

        from seapilot_agent_qa import SeaPilotAgent
    else:
        print("🛠️ No '?' detected -> Routing to **Operation Agent** (seapilot_agent_op)")

        from seapilot_agent_op import SeaPilotAgent

    agent = SeaPilotAgent(env)

    max_steps = int(task.complexity * 10) 
    print(f"⏳ Max Steps limit: {max_steps} (Standard Action Budget)")

    step_counter = 0
    task_success = False
    final_score = 0.0
    
    try:
        while step_counter < max_steps:
            print(f"\n--- Step {step_counter+1} / {max_steps} ---")
            
            res = agent.step(task.goal)
            time.sleep(1)

            if res.done:
                print("🤖 Agent reported DONE.")
                final_score = task.is_successful(env)
                
                if final_score == 1.0:
                    print(f"✅ Task Successfully Completed! (Score: {final_score})")
                    task_success = True
                elif final_score > 0.0:
                    print(f"⚠️ Task Partially Completed. (Score: {final_score})")
                else:
                    print(f"❌ Task Failed (Agent thought it was done, but Score is {final_score}).")

                if not res.data.get("skip_cost", False):
                    step_counter += 1
                break
                
            is_last_subgoal = res.data.get("is_last_subgoal", False)
            if is_last_subgoal:
                print("🔍 [System] In Last Subgoal Phase -> Checking Environment Success...")
                current_score = task.is_successful(env)
                
                if current_score == 1.0:
                    print(f"✅ Task Successfully Completed (Detected early)! (Score: {current_score})")
                    task_success = True
                    final_score = current_score
                    if not res.data.get("skip_cost", False):
                        step_counter += 1
                    break
                elif current_score > 0.0:
                    print(f"   (Current Score: {current_score} - Partially done, continuing...)")
                else:
                    print("   (Environment check failed, Score: 0.0, continuing...)")

            if not res.data.get("skip_cost", False):
                step_counter += 1
            else:
                print("👀 [System] Monitor Step - Free of charge (Counter paused)")
                
        if not task_success and step_counter >= max_steps:
            final_score = task.is_successful(env)
            print(f"❌ Task Failed (Max steps reached). Final Score: {final_score}")

    finally:
        print("\n🧹 Tearing down task state...")
        if hasattr(agent, "rule_manager") and hasattr(agent, "active_rule_ids"):
            try:
                if agent.active_rule_ids:
                    agent.rule_manager.update_stats(agent.active_rule_ids, success=bool(task_success))
            except Exception as e:
                print(f"[RuleManager Warning] stats update failed: {e}")

        # Print META_ report BEFORE teardown so it is always written even if teardown fails.
        m = agent.metrics
        real_system_overhead = m.edge_total_latency - (m.monitor_latency + m.executor_latency + m.cloud_latency)
        if real_system_overhead < 0: real_system_overhead = 0.0

        print("\n" + "#"*60)
        print(f"## 📊 AUTOMATION METRICS REPORT :: {FLAGS.task} ##")
        print("#"*60)

        print(f">> META_Task_Name   : {FLAGS.task}")
        agent_type = "QA" if "?" in final_goal_text else "OP"
        print(f">> META_Agent_Type  : {agent_type}")
        print(f">> META_Status      : {'SUCCESS' if task_success else 'FAILURE'}")
        print(f">> META_Final_Score : {final_score:.1f}")
        print(f">> META_Action_Steps: {step_counter}")
        print(f">> META_Replan_Count: {agent.replan_count}")

        print("-" * 60)
        print(f">> PERF_Cloud       : Tokens={m.cloud_tokens:<6} | Latency={m.cloud_latency:.4f}s")
        print(f">> PERF_Monitor     : Tokens={m.monitor_tokens:<6} | Latency={m.monitor_latency:.4f}s")
        print(f">> PERF_Executor    : Tokens={m.executor_tokens:<6} | Latency={m.executor_latency:.4f}s")
        print(f">> PERF_System      : Overhead={real_system_overhead:.4f}s")
        print("-" * 60)
        print(f">> Total   : All Steps={m.edge_total_steps:<5}  | Latency={m.edge_total_latency:.4f}s")
        print("#"*60 + "\n")

        task.tear_down(env)
        env.close()
        print("🏁 Environment closed.")

if __name__ == '__main__':
    app.run(main)