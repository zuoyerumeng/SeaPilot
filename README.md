# SeaPilot: Mobile Agent with Self-refining Environment Alignment

This repository contains the anonymized implementation of **SeaPilot**, a hierarchical agent architecture designed for Android automation tasks. The implementation is built upon the [AndroidWorld](https://github.com/google-research/android_world) environment.

---

## 📂 Repository Structure & Functionality

This codebase consists of core components designed to handle different aspects of interaction:

- **`seapilot_agent_op.py`**  
  Implements the architecture and prompts for **Operation-Oriented Tasks** (pure UI manipulation without information extraction).

- **`seapilot_agent_qa.py`**  
  Implements the architecture and prompts for **QA-Oriented Tasks** (information retrieval and question answering).

- **`run_seapilot.py`**  
  Main entry point for the architecture.
  - **Dynamic Routing**: Automatically detects the task type and routes execution to either the OP agent or QA agent.
  - **Logging**: Generates detailed execution logs for each task.

- **`analyze_logs.py`**  
  Evaluation script used to calculate metrics and success rates across the 116 AndroidWorld task categories based on generated logs.

- **`rule_manager.py`**  
  Rule retrieval and selection module used by both OP/QA agents, including staged rule filtering and online rule statistics update.

---

## 🚀 Installation & Setup

Since SeaPilot relies on the `android_world` library, you must set up the base environment first and then integrate the agents.

### 1. Prerequisites

- **Python**: 3.11.8 (recommended)
- **ADB**: Ensure `adb` is installed and available in your system `PATH`
- **Android Environment**: Android emulator (AVD) or a physical device connected via ADB

### 2. Install AndroidWorld

Follow the official installation guide from the AndroidWorld repository

### 3. Integrate SeaPilot Files

Move the files from this package into the installed `android_world` directory structure as follows:

#### 3.1 Move Agent Files

Copy the agent implementations into the AndroidWorld agents directory:

```bash
cp seapilot_agent_op.py /path/to/android_world/android_world/agents/
cp seapilot_agent_qa.py /path/to/android_world/android_world/agents/
```

#### 3.2 Move Script Files

Copy the main runner and evaluation scripts to the AndroidWorld root directory:

```bash
cp run_seapilot.py /path/to/android_world/
cp analyze_logs.py /path/to/android_world/
```

#### 3.3 Move Rule Manager File

Copy the rule manager implementation into the AndroidWorld rules directory:

```bash
cp rule_manager.py /path/to/android_world/android_world/agents/rules/
```

---

## ⚙️ Configuration

To run the architecture successfully, configure both the Android emulator connection and internal agent settings.

### Emulator Connection Setup (`run_seapilot.py`)

The main entry script requires the following parameters:

- `--adb_path`: Path to the ADB executable
- `--console_port`: Port of the Android emulator

### Agent Logic Configuration (`seapilot_agent_op.py` & `seapilot_agent_qa.py`)

Open both files located in:

```text
android_world/android_world/agents/
```

Configure the following:

- **LLM Setup**: Configure model selection and runtime settings
- **Swipe Action Setup**: Set the `adb_bin` variable
  - This is required for executing physical swipe actions (e.g., brightness adjustment) via `subprocess`
- **Rule Setup**: Ensure `rule_manager.py` and `rules.json` are both located under `android_world/android_world/agents/rules/`

---

## 🏃 Usage

### Running a Single Task

Execute the hierarchical agent on a specific AndroidWorld task from the `android_world` root directory.

**Command format:**

```bash
python run_seapilot.py --task=<TaskName>
```

**Example (Clock task):**

```bash
python run_seapilot.py --task=ClockStopWatchPausedVerify
```

---

## 📊 Evaluating Results

After running tasks, logs will be generated in the `results/` directory. To compute success rates and metrics:

```bash
python analyze_logs.py
```

---

## ⚠️ Anonymity Statement

This repository is anonymized for peer review. All author names, affiliations, and institutional details have been removed.
