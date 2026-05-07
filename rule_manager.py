"""
Rule Manager for AdecPilot Hierarchical Agent.

Provides structured rule storage (rules.json) with:
  - Semantic/keyword-based candidate retrieval
  - Thompson Sampling (Beta distribution) for rule selection
  - Stats tracking (uses / successes) for online learning

Usage:
    from android_world.agents.rules.rule_manager import RuleManager
    import openai

    client = openai.OpenAI(api_key=..., base_url=...)
    rm = RuleManager(edge_text_client=client, model_name="Qwen3-4B-Instruct")

    guidance = rm.get_planning_guidance(task_goal, scope="op")
    monitor_rules = rm.get_monitoring_rules(task_goal)
    rm.update_stats(rm.last_selected_ids, success=True)
"""

import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

# Default path relative to this file
_DEFAULT_RULES_PATH = Path(__file__).parent / "rules.json"


# ─────────────────────────────────────────────────
# Module-level I/O helpers
# ─────────────────────────────────────────────────

def load_rules(path: Optional[Path] = None) -> List[Dict]:
    p = path or _DEFAULT_RULES_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[RuleManager] WARNING: Could not load rules from {p}: {e}")
        return []


def save_rules(rules: List[Dict], path: Optional[Path] = None) -> None:
    p = path or _DEFAULT_RULES_PATH
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[RuleManager] WARNING: Could not save rules to {p}: {e}")


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    return [str(value).strip()] if str(value).strip() else []


def _compute_confidence(rule: Dict) -> float:
    uses = max(int(rule.get("uses", 0) or 0), 0)
    successes = max(int(rule.get("successes", 0) or 0), 0)
    successes = min(successes, uses)
    return (successes + 1) / (uses + 2)


def _normalize_rule(rule: Dict) -> Dict:
    """Backfill structured fields and confidence while keeping old schema usable."""
    r = dict(rule)
    r["uses"] = max(int(r.get("uses", 0) or 0), 0)
    r["successes"] = max(int(r.get("successes", 0) or 0), 0)
    if r["successes"] > r["uses"]:
        r["successes"] = r["uses"]

    structured = r.get("structured", {})
    if not isinstance(structured, dict):
        structured = {}

    applicable_when = _to_list(structured.get("applicable_when"))
    environment_constraints = _to_list(structured.get("environment_constraints"))
    operation_suggestions = _to_list(structured.get("operation_suggestions"))

    # Backward-compatible migration from legacy content.
    if not operation_suggestions:
        legacy = str(r.get("content", "")).strip()
        if legacy:
            operation_suggestions = [legacy]

    if not applicable_when:
        tags = _to_list(r.get("tags"))
        if tags:
            applicable_when = [f"Task relates to: {', '.join(tags[:6])}"]

    if not environment_constraints:
        pattern = str(r.get("task_pattern", "")).strip()
        if pattern:
            environment_constraints = [f"Match task regex: {pattern}"]

    r["structured"] = {
        "applicable_when": applicable_when,
        "environment_constraints": environment_constraints,
        "operation_suggestions": operation_suggestions,
    }
    r["confidence"] = round(_compute_confidence(r), 6)
    return r


# ─────────────────────────────────────────────────
# Thompson Sampling helper
# ─────────────────────────────────────────────────

def _thompson_sample(candidates: List[Dict], max_select: int) -> List[Dict]:
    """Beta-distribution Thompson Sampling.

    Draws a sample θ ~ Beta(successes+1, failures+1) for each candidate,
    then picks the top-k by drawn θ.
    """
    if not candidates:
        return []
    scored = []
    for r in candidates:
        uses = max(int(r.get("uses", 0) or 0), 0)
        successes = max(int(r.get("successes", 0) or 0), 0)
        successes = min(successes, uses)
        failures = uses - successes
        theta = random.betavariate(successes + 1, failures + 1)
        scored.append((theta, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:max_select]]


# ─────────────────────────────────────────────────
# Env-var helpers
# ─────────────────────────────────────────────────

def _get_env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def _get_env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default
    except Exception:
        return default


def _get_env_list(name: str) -> List[str]:
    v = os.environ.get(name, "")
    if not v:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


# ─────────────────────────────────────────────────
# Stage-1 LLM filter prompt
# ─────────────────────────────────────────────────

_STAGE1_PROMPT = """You are a rule selector for an Android automation agent.

Task goal:
{task_goal}

Below are candidate rules. Return a JSON array of rule IDs that are **directly applicable** to this task.
Only include rules whose guidance would help the agent succeed at this specific goal.
If no rule is applicable, return an empty array [].

Rules:
{rules_json}

Respond ONLY with a JSON array of rule IDs, e.g.: ["rule_op_003", "rule_qa_002"]
"""

_STAGE1_WEAK_PROMPT = """You are a rule selector for an Android automation agent.

Task goal:
{task_goal}

Below are candidate rules. Return a JSON array of rule IDs that *might* be loosely related.
Prefer to return more rules rather than fewer, even if the match is approximate.

Rules:
{rules_json}

Respond ONLY with a JSON array of rule IDs, e.g.: ["rule_op_003", "rule_qa_002"]
"""


# ─────────────────────────────────────────────────
# RuleManager
# ─────────────────────────────────────────────────

class RuleManager:
    """Central rule management with retrieval, Stage1 LLM filter, Stage2 Thompson Sampling."""

    def __init__(
        self,
        edge_text_client=None,
        model_name: str = "Qwen3-4B-Instruct",
        rules_path: Optional[Path] = None,
    ):
        # Allow overriding rules path via env var (for ablation experiments).
        env_rules_path = os.environ.get("RULES_PATH", "").strip()
        self.rules_path = Path(env_rules_path) if env_rules_path else (rules_path or _DEFAULT_RULES_PATH)

        raw_rules = load_rules(self.rules_path)
        # Deduplicate by ID (keep last occurrence to preserve later updates).
        seen: Dict[str, Dict] = {}
        for r in raw_rules:
            rid = r.get("id")
            if rid:
                seen[rid] = r
        self.rules: List[Dict] = [_normalize_rule(r) for r in seen.values()]

        self.edge_text_client = edge_text_client
        self.model_name = model_name

        # Runtime configuration from environment variables.
        self.scope_round = _get_env_int("RULE_SCOPE_ROUND", 5)
        self.stage1_enabled = _get_env_bool("RULE_STAGE1_ENABLED", True)
        self.stage2_enabled = _get_env_bool("RULE_STAGE2_ENABLED", True)
        self.stage1_weak = _get_env_bool("RULE_STAGE1_WEAK", False)
        self.stage2_mode = str(os.environ.get("RULE_STAGE2_MODE", "thompson")).strip().lower()
        # Hyperparameter overrides for ablation only. Keep defaults when unset.
        self.retrieval_k = _get_env_int("RULE_RETRIEVAL_K", 0)
        self.select_m = _get_env_int("RULE_SELECT_M", 0)
        # Optional forced rules for controlled ablations.
        self.force_rule_ids = _get_env_list("RULE_FORCE_IDS")

        # Tracks IDs of rules selected in the most recent call.
        self.last_selected_ids: List[str] = []

        print(
            f"[RuleManager] Loaded {len(self.rules)} rules from {self.rules_path}. "
            f"round={self.scope_round}, stage1={self.stage1_enabled}, "
            f"stage1_weak={self.stage1_weak}, stage2={self.stage2_enabled}, "
            f"stage2_mode={self.stage2_mode}, k={self.retrieval_k or 'default'}, "
            f"m={self.select_m or 'default'}, force_ids={len(self.force_rule_ids)}"
        )

    # ── Candidate retrieval (keyword/regex matching) ──────────────────────

    def retrieve_candidates(
        self,
        task_goal: str,
        phase: str = "planning",
        scope: str = "op",
        k: int = 12,
    ) -> List[Dict]:
        """Return up to k candidate rules matching task_goal by keyword/regex."""
        goal_l = (task_goal or "").lower()
        # Determine which scope tokens are permitted.
        scope_map = {
            "op": {"op", "both", ""},
            "qa": {"qa", "both", ""},
            "mon": {"mon", "op", "both", ""},  # monitoring rules can come from op scope too
        }
        allowed_scopes = scope_map.get(scope, {"both", ""})

        # Phase → scope token mapping in rule data.
        phase_map = {
            "planning": {"op", "qa"},
            "monitoring": {"mon"},
        }
        # For planning phase, filter by agent scope (op/qa).
        # For monitoring phase, filter rules with scope "mon" or matching.

        results: List[Dict] = []
        for r in self.rules:
            r_scope = str(r.get("scope", "")).lower()
            r_id = r.get("id", "")

            # Phase-based filtering.
            if phase == "planning":
                # op rules for op scope, qa rules for qa scope, scope="" (generic) always ok
                if r_scope not in allowed_scopes and r_scope != "":
                    continue
                # Skip monitoring rules for planning.
                if r_id.startswith("rule_mon_"):
                    continue
            elif phase == "monitoring":
                # Only include monitoring rules.
                if not r_id.startswith("rule_mon_"):
                    continue

            # Task pattern matching.
            pattern = str(r.get("task_pattern", "")).strip()
            if pattern:
                try:
                    if not re.search(pattern, goal_l, re.IGNORECASE):
                        continue
                except re.error:
                    # Fallback: simple substring check on pattern parts.
                    parts = [p.strip() for p in re.split(r"[|()]", pattern) if p.strip() and len(p.strip()) > 2]
                    if not any(p in goal_l for p in parts):
                        continue

            results.append(r)
            if len(results) >= k * 2:
                break

        # Sort by confidence descending.
        results.sort(key=lambda x: -x.get("confidence", 0.5))
        return results[:k]

    # ── Scope-round filter ────────────────────────────────────────────────

    def _apply_scope_round(self, task_goal: str, candidates: List[Dict]) -> List[Dict]:
        """If scope_round == 0, return empty (disable all rules).
        Otherwise filter candidates to those with creation_round <= scope_round.
        Rules without a creation_round field default to 5 (available only at full history).
        """
        if self.scope_round <= 0:
            return []
        return [r for r in candidates if r.get("creation_round", 5) <= self.scope_round]

    # ── Forced rules (for ablation experiments) ──────────────────────────

    def _collect_forced_rules(
        self,
        task_goal: str,
        phase: str,
        scope: str,
        max_rules: int,
    ) -> List[Dict]:
        if not self.force_rule_ids:
            return []
        id_set = set(self.force_rule_ids)
        goal_l = (task_goal or "").lower()
        selected: List[Dict] = []
        for r in self.rules:
            if r.get("id") not in id_set:
                continue
            # Respect task regex if provided.
            pat = (r.get("task_pattern") or "").strip()
            if pat:
                try:
                    if not re.search(pat, goal_l, re.IGNORECASE):
                        continue
                except re.error:
                    continue
            selected.append(r)
            if len(selected) >= max_rules:
                break
        return selected

    # ── Stage-1 LLM filter ────────────────────────────────────────────────

    def _stage1_filter(
        self,
        task_goal: str,
        candidates: List[Dict],
        max_select: int,
    ) -> List[Dict]:
        """Use edge LLM to filter/rank candidates. Falls back to candidates on error."""
        if not self.edge_text_client or not candidates:
            return candidates

        prompt_template = _STAGE1_WEAK_PROMPT if self.stage1_weak else _STAGE1_PROMPT
        rules_preview = []
        for r in candidates:
            structured = r.get("structured", {})
            suggestions = structured.get("operation_suggestions", [])
            preview = suggestions[0][:200] if suggestions else str(r.get("content", ""))[:200]
            rules_preview.append({"id": r["id"], "name": r.get("name", ""), "summary": preview})

        prompt = prompt_template.format(
            task_goal=task_goal,
            rules_json=json.dumps(rules_preview, ensure_ascii=False, indent=2),
        )

        try:
            resp = self.edge_text_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse JSON array from response.
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                selected_ids = json.loads(m.group())
                if isinstance(selected_ids, list):
                    id_set = set(str(x) for x in selected_ids)
                    filtered = [r for r in candidates if r.get("id") in id_set]
                    if filtered:
                        return filtered[:max_select]
        except Exception as e:
            print(f"[RuleManager] Stage1 LLM error: {e}")

        # Fallback: return top candidates by confidence.
        return candidates[:max_select]

    # ── Public: get_planning_guidance ─────────────────────────────────────

    def get_planning_guidance(
        self,
        task_goal: str,
        scope: str = "op",
        max_rules: int = 6,
    ) -> str:
        """Return planning guidance string for the cloud planner."""
        select_m = self.select_m if self.select_m > 0 else max_rules
        retrieval_k = self.retrieval_k if self.retrieval_k > 0 else (select_m * 2)
        candidates = self.retrieve_candidates(
            task_goal, phase="planning", scope=scope, k=retrieval_k
        )
        candidates = self._apply_scope_round(task_goal, candidates)

        # Prepend forced rules if configured.
        forced = self._collect_forced_rules(
            task_goal, phase="planning", scope=scope, max_rules=select_m
        )
        if forced:
            seen = {r["id"] for r in forced}
            merged = list(forced)
            for r in candidates:
                if r["id"] in seen:
                    continue
                merged.append(r)
                if len(merged) >= select_m:
                    break
            candidates = merged

        if not candidates:
            self.last_selected_ids = []
            return ""

        # Stage 1: LLM filter.
        if self.stage1_enabled and self.edge_text_client:
            candidates = self._stage1_filter(task_goal, candidates, max_select=select_m)

        # Stage 2: Thompson Sampling or top-k.
        if self.stage2_enabled:
            if self.stage2_mode == "top1_sr":
                candidates.sort(key=lambda r: -r.get("confidence", 0.5))
                selected = candidates[:1]
            else:
                selected = _thompson_sample(candidates, max_select=select_m)
        else:
            selected = candidates[:select_m]

        self.last_selected_ids = [r["id"] for r in selected]

        if not selected:
            return ""

        # Format as guidance text.
        lines = ["**RULE GUIDANCE (apply if applicable):**"]
        for i, r in enumerate(selected, 1):
            structured = r.get("structured", {})
            suggestions = structured.get("operation_suggestions", [])
            when = structured.get("applicable_when", [])
            name = r.get("name", r.get("id", ""))
            when_str = when[0] if when else ""
            sugg_str = suggestions[0] if suggestions else str(r.get("content", ""))
            entry = f"{i}. [{name}]"
            if when_str:
                entry += f" (When: {when_str})"
            if sugg_str:
                entry += f": {sugg_str}"
            lines.append(entry)

        return "\n".join(lines)

    # ── Public: get_monitoring_rules ──────────────────────────────────────

    def get_monitoring_rules(self, task_goal: str, max_rules: int = 3) -> str:
        """Return monitoring rules string for the visual monitor."""
        candidates = self.retrieve_candidates(
            task_goal, phase="monitoring", scope="op", k=max_rules * 2
        )
        candidates = self._apply_scope_round(task_goal, candidates)

        # Prepend forced monitoring rules if configured.
        forced = self._collect_forced_rules(
            task_goal, phase="monitoring", scope="op", max_rules=max_rules
        )
        if forced:
            seen = {r["id"] for r in forced}
            merged = list(forced)
            for r in candidates:
                if r["id"] in seen:
                    continue
                merged.append(r)
                if len(merged) >= max_rules:
                    break
            candidates = merged

        if not candidates:
            return ""

        # Stage 2 only for monitoring (no Stage1 LLM for speed).
        if self.stage2_enabled:
            if self.stage2_mode == "top1_sr":
                candidates.sort(key=lambda r: -r.get("confidence", 0.5))
                selected = candidates[:1]
            else:
                selected = _thompson_sample(candidates, max_select=max_rules)
        else:
            selected = candidates[:max_rules]

        if not selected:
            return ""

        lines = ["**MONITORING RULES:**"]
        for i, r in enumerate(selected, 1):
            structured = r.get("structured", {})
            suggestions = structured.get("operation_suggestions", [])
            name = r.get("name", r.get("id", ""))
            sugg_str = suggestions[0] if suggestions else str(r.get("content", ""))
            lines.append(f"{i}. [{name}]: {sugg_str}")

        return "\n".join(lines)

    # ── Public: update_stats ──────────────────────────────────────────────

    def update_stats(self, rule_ids: List[str], success: bool) -> None:
        """Increment uses (and optionally successes) for given rule IDs."""
        if not rule_ids:
            return
        id_set = set(rule_ids)
        changed = False
        for r in self.rules:
            if r.get("id") in id_set:
                r["uses"] = max(int(r.get("uses", 0) or 0), 0) + 1
                if success:
                    r["successes"] = max(int(r.get("successes", 0) or 0), 0) + 1
                r["confidence"] = round(_compute_confidence(r), 6)
                changed = True
        if changed:
            try:
                save_rules(self.rules, self.rules_path)
            except Exception as e:
                print(f"[RuleManager] WARNING: Could not save updated stats: {e}")
