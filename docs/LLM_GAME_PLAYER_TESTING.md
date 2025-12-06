# LLM Game Player Testing with Local GPU Enhanced Execution
**Benchmark Framework for: ModelForecast**
**Date**: 2025-12-05
**Focus**: LLM↔GPU Communication Architecture | GPU-Accelerated Game Simulation

## Executive Summary

This document analyzes intervention protocol design for an LLM benchmark that runs GPU-simulated games (DOOM, Angband, CoreWars). LLMs have three roles: **TUNER** (pre-game config), **OBSERVER** (state monitoring), **INTERVENOR** (mid-game overrides). The core challenge: GPU simulation runs at 4-40M ticks/sec, while LLM API calls take 200-2000ms with per-call costs.

**Key Finding**: Hybrid event-driven + checkpoint protocol with async simulation enables 90%+ GPU utilization while maintaining meaningful LLM control.

---

## 1. Intervention Granularity Analysis

### 1.1 Options Comparison

| Approach | API Calls/Game | Latency Impact | GPU Utilization | Cost (1K games) | Use Case |
|----------|---------------|----------------|-----------------|-----------------|----------|
| **Every Turn** | 10K-100K | Catastrophic | <1% | $500-5000 | Research only |
| **Event-Triggered** | 50-500 | Moderate | 60-80% | $25-250 | Balanced |
| **Periodic Checkpoints** | 10-100 | Low | 85-95% | $5-50 | Production |
| **Hybrid (Events+Checkpoints)** | 30-300 | Low-Moderate | 70-90% | $15-150 | **Recommended** |
| **"Wake Me When X"** | 5-50 | Very Low | 90-98% | $2.5-25 | Autonomous w/ supervision |

### 1.2 Game-Specific Patterns

#### DOOM (Real-time FPS)
- **Base tick rate**: 35 Hz (28ms/tick)
- **GPU simulation**: 100K instances × 1000 ticks in 25ms (140M ticks/sec)
- **Event candidates**: Health <30%, ammo depleted, monster encountered, key found, door blocked
- **Checkpoint interval**: Every 35 ticks (1 second game-time) or every 10 kills
- **Typical game**: 300 ticks (8.5s) → 9 checkpoints + 5-15 events = **14-24 LLM calls**

#### Angband (Turn-based Roguelike)
- **Base tick rate**: Varies by player speed (1-3 player actions/sec typical)
- **GPU simulation**: 10K instances × 1000 turns in 2.5s (4M turns/sec)
- **Event candidates**: HP <25%, monster unique spotted, artifact found, depth milestone (every 5 levels), critical combat (monster HP >player HP)
- **Checkpoint interval**: Every 50 turns or every level transition
- **Typical game**: 10,000 turns (death at depth 15) → 200 checkpoints + 30-80 events = **230-280 LLM calls**

#### CoreWars (Programming Competition)
- **Base tick rate**: MARS executes at 8000 cycles/match max
- **GPU simulation**: 1024 instances × 8000 cycles in 0.2s (40M cycles/sec)
- **Event candidates**: Warrior code overwritten, execution pointer captured, warrior eliminated
- **Checkpoint interval**: Every 1000 cycles
- **Typical match**: 3000 cycles (avg) → 3 checkpoints + 2-5 events = **5-8 LLM calls**
- **Note**: LLM intervention is TUNER-only (pre-match warrior generation). Real-time intervention not meaningful for code battles.

### 1.3 Hybrid Protocol Design

**Principle**: Events catch critical moments, checkpoints catch gradual changes.

```python
class InterventionTrigger:
    def __init__(self, game_type: str):
        self.events = self._init_events(game_type)
        self.checkpoint_interval = self._init_checkpoint(game_type)
        self.last_checkpoint = 0

    def should_intervene(self, state: GameState) -> tuple[bool, str]:
        # Priority 1: Critical events
        for event in self.events:
            if event.condition(state):
                return (True, f"event:{event.name}")

        # Priority 2: Periodic checkpoint
        if state.tick - self.last_checkpoint >= self.checkpoint_interval:
            self.last_checkpoint = state.tick
            return (True, f"checkpoint:{state.tick}")

        return (False, "")
```

**Example - Angband Critical Event**:
```cuda
// Device-side event detection
__device__ bool check_critical_event(BorgState* s, int* event_type) {
    // HP critical (below 25%)
    if (s->hp < s->max_hp / 4) {
        *event_type = EVENT_HP_CRITICAL;
        return true;
    }

    // Monster too strong (expected damage > current HP)
    int danger = calculate_danger(s);
    if (danger > s->hp) {
        *event_type = EVENT_LETHAL_DANGER;
        return true;
    }

    // Unique monster spotted
    for (int m = 0; m < s->monster_count; m++) {
        if (s->monster_type[m] >= MONSTER_UNIQUE_START) {
            *event_type = EVENT_UNIQUE_SPOTTED;
            return true;
        }
    }

    return false;
}
```

---

## 2. Action Space Per Game

### 2.1 Abstraction Level Trade-offs

| Level | Granularity | Examples | Pros | Cons |
|-------|-------------|----------|------|------|
| **Atomic** | Individual keypresses/commands | DOOM: `+forward`, `+fire`, `+strafeleft` | Precise control | High frequency, verbose |
| **Tactical** | Semantic actions | DOOM: `attack(target_id)`, `navigate(x,y)`, `use_item(med_kit)` | Balanced | Requires game-side interpretation |
| **Strategic** | High-level directives | DOOM: `clear_room()`, `find_exit()`, `avoid_combat()` | Low frequency, intuitive | Less control, complex implementation |
| **Hybrid** | Context-dependent mixing | Critical: atomic, routine: strategic | **Optimal** | Requires sophisticated protocol |

### 2.2 Game-Specific Action Spaces

#### DOOM Action Protocol

**Atomic Actions** (19 actions):
```python
ATOMIC_ACTIONS = {
    # Movement (8)
    "forward": BT_FORWARD,
    "back": BT_BACK,
    "strafe_left": BT_STRAFE_LEFT,
    "strafe_right": BT_STRAFE_RIGHT,
    "turn_left": BT_TURN_LEFT,
    "turn_right": BT_TURN_RIGHT,
    "turn_to(angle)": BT_TURN_TO,
    "move_to(x, y)": BT_MOVE_TO,

    # Combat (4)
    "fire": BT_ATTACK,
    "switch_weapon(id)": BT_CHANGE,
    "aim_at(target_id)": BT_AIM,
    "melee_attack": BT_MELEE,

    # Interaction (4)
    "use": BT_USE,  # Doors, switches
    "pickup": BT_PICKUP,
    "toggle_run": BT_RUN,
    "sprint": BT_SPRINT,

    # Meta (3)
    "wait": BT_WAIT,
    "save_ammo": BT_CONSERVE,
    "retreat": BT_RETREAT,
}
```

**Tactical Macros** (combines atomics):
```python
TACTICAL_MACROS = {
    "circle_strafe_left": lambda: [
        {"action": "strafe_left", "duration": 4},
        {"action": "fire", "duration": 1},
        {"action": "turn_left", "angle": 15}
    ],

    "retreat_firing": lambda target: [
        {"action": "aim_at", "target": target},
        {"action": "fire", "duration": 2},
        {"action": "back", "duration": 5}
    ],

    "door_breach": lambda: [
        {"action": "use"},  # Open door
        {"action": "back", "duration": 2},  # Step back
        {"action": "switch_weapon", "weapon": "shotgun"},
        {"action": "forward", "duration": 3},
        {"action": "fire", "duration": 1}
    ]
}
```

**Strategic Directives** (AI-assisted pathfinding):
```python
STRATEGIC_DIRECTIVES = {
    "explore_area": {
        "impl": "pathfind_to_unexplored",
        "params": ["avoid_combat: bool", "prioritize: health|ammo|exit"]
    },

    "eliminate_threats": {
        "impl": "engage_all_visible",
        "params": ["priority: closest|weakest|strongest", "fallback: retreat|hold"]
    },

    "find_objective": {
        "impl": "navigate_to_objective",
        "params": ["objective: keycard|exit|weapon", "timeout_turns: int"]
    }
}
```

**LLM Intervention Protocol**:
```json
{
    "game": "doom",
    "instance_id": 42,
    "tick": 350,
    "trigger": "event:hp_critical",
    "state": {
        "health": 28,
        "armor": 10,
        "ammo": {"bullets": 45, "shells": 8},
        "weapon": "shotgun",
        "position": {"x": 1200, "y": -3600},
        "visible_monsters": [
            {"type": "zombie", "distance": 180, "hp": 30},
            {"type": "imp", "distance": 350, "hp": 60}
        ],
        "nearby_items": [
            {"type": "medikit", "distance": 240}
        ]
    },
    "llm_decision": {
        "action_type": "tactical_macro",
        "macro": "retreat_to_health",
        "parameters": {
            "health_item_id": 7,
            "cover_position": {"x": 1100, "y": -3650}
        },
        "duration": 35  // ticks to execute before next check
    }
}
```

#### DOOM Reference Implementations

This benchmark builds on established DOOM AI research infrastructure:

**ViZDoom** ([external/ViZDoom](../external/ViZDoom) | [Farama Foundation](https://github.com/Farama-Foundation/ViZDoom))
- Standard API for DOOM reinforcement learning research
- Python bindings to DOOM engine with Gymnasium integration
- Provides `GameVariables` (health, ammo, kills) that map directly to our JSON protocol
- Supports custom scenarios via WAD files
- Our GPU simulation diverges from ViZDoom's rendering-based approach: we simulate game logic at 140M ticks/sec vs ViZDoom's frame-rate-limited execution

**Arnold** ([external/Arnold](../external/Arnold) | [Guillaume Lample, 2016](https://github.com/glample/Arnold))
- Landmark DRL implementation: "Playing FPS Games with Deep Reinforcement Learning"
- First to demonstrate navigation + combat in DOOM deathmatch
- Architecture: Separate navigation and action networks with curriculum learning
- Uses ViZDoom as backend
- Serves as baseline for comparing LLM game-playing against trained DRL agents

**Protocol Compatibility**:
| Our Protocol Field | ViZDoom GameVariable | Arnold Feature |
|-------------------|---------------------|----------------|
| `state.health` | `HEALTH` | `game_feature[0]` |
| `state.ammo` | `AMMO*` | `game_feature[1-4]` |
| `state.position` | `POSITION_X/Y` | Navigation network input |
| `visible_monsters` | Depth buffer + labels | Object detection layer |

#### Angband Action Protocol

**Atomic Actions** (28 core commands):
```python
ANGBAND_ATOMIC = {
    # Movement (10)
    "move_north": BORG_ACTION_MOVE_N,
    "move_south": BORG_ACTION_MOVE_S,
    "move_east": BORG_ACTION_MOVE_E,
    "move_west": BORG_ACTION_MOVE_W,
    "move_ne": BORG_ACTION_MOVE_NE,
    "move_nw": BORG_ACTION_MOVE_NW,
    "move_se": BORG_ACTION_MOVE_SE,
    "move_sw": BORG_ACTION_MOVE_SW,
    "rest": BORG_ACTION_REST,
    "descend": BORG_ACTION_DESCEND,

    # Combat (6)
    "attack(direction)": BORG_ACTION_ATTACK,
    "fire_missile(target)": BORG_ACTION_FIRE,
    "cast_spell(spell_id)": BORG_ACTION_CAST,
    "use_wand(wand_id, target)": BORG_ACTION_USE_WAND,
    "read_scroll(scroll_id)": BORG_ACTION_READ,
    "quaff_potion(potion_id)": BORG_ACTION_QUAFF,

    # Utility (12)
    "teleport": BORG_ACTION_TELEPORT,
    "heal": BORG_ACTION_HEAL,
    "detect_monsters": BORG_ACTION_DETECT,
    "identify_item(item_id)": BORG_ACTION_IDENTIFY,
    "drop_item(item_id)": BORG_ACTION_DROP,
    "pickup_item": BORG_ACTION_PICKUP,
    "wear_armor(item_id)": BORG_ACTION_WEAR,
    "wield_weapon(item_id)": BORG_ACTION_WIELD,
    "enchant_weapon": BORG_ACTION_ENCHANT,
    "recall_to_town": BORG_ACTION_RECALL,
    "explore": BORG_ACTION_EXPLORE,
    "flee": BORG_ACTION_FLEE,
}
```

**Strategic Overrides** (for Borg autopilot):
```python
BORG_OVERRIDES = {
    "force_retreat": {
        "description": "Override borg combat decision, force teleport/recall",
        "impl": "immediate_action_override",
        "priority": "critical"
    },

    "halt_descent": {
        "description": "Prevent descending below current depth",
        "impl": "constraint_addition",
        "duration": "until_revoked"
    },

    "hunt_unique(monster_id)": {
        "description": "Prioritize finding/killing specific unique monster",
        "impl": "goal_injection",
        "duration": "until_killed_or_depth_change"
    },

    "avoid_stairs": {
        "description": "Don't take stairs (up or down) for N turns",
        "impl": "action_blacklist",
        "duration": "N_turns"
    }
}
```

**LLM Intervention Protocol**:
```json
{
    "game": "angband",
    "instance_id": 123,
    "turn": 850,
    "trigger": "event:unique_spotted",
    "state": {
        "depth": 15,
        "hp": 85,
        "max_hp": 120,
        "sp": 12,
        "max_sp": 45,
        "clevel": 18,
        "position": {"x": 45, "y": 23},
        "visible_monsters": [
            {"name": "Smeagol", "unique": true, "hp": 150, "distance": 5},
            {"name": "Orc warrior", "hp": 40, "distance": 3}
        ],
        "inventory": {
            "healing_potions": 3,
            "teleport_scrolls": 2,
            "detect_monster_scrolls": 1
        },
        "borg_plan": "ATTACK"  // What borg WOULD do
    },
    "llm_decision": {
        "override": true,
        "action": "strategic_directive",
        "directive": "retreat_and_prepare",
        "reason": "Smeagol too dangerous at current level, need buffs",
        "actions": [
            {"action": "quaff_potion", "item": "blessing"},
            {"action": "read_scroll", "item": "teleport"},
            {"action": "halt_descent", "until": "clevel >= 20"}
        ]
    }
}
```

#### CoreWars - TUNER Only (No Runtime Intervention)

CoreWars intervention is **pre-game only** (warrior generation). Runtime intervention not meaningful because:
1. Matches last 8000 cycles (0.2s on GPU)
2. Warriors are self-executing code (no "player" to control)
3. Modification requires rewriting MARS core memory (breaks competitive rules)

**LLM Role**: Generate warrior code, not intervene during battles.

```json
{
    "game": "corewars",
    "phase": "tuner",
    "tournament_round": 5,
    "performance_feedback": {
        "warrior_name": "bomber_v3",
        "wins": 12,
        "losses": 28,
        "ties": 10,
        "win_rate": 0.24,
        "common_defeat_cause": "imp_spiral",
        "avg_cycles_survived": 2300
    },
    "llm_decision": {
        "action": "generate_improved_warrior",
        "strategy_adjustment": "Add imp-detection and bomber retaliation",
        "warrior_code": ";name Bomber_v4\n;author LLM\nDAT.F #0, #0\n..."
    }
}
```

---

## 3. Borg Override Mechanics (Angband Specific)

### 3.1 Borg Autonomy Spectrum

```
Full Autonomy ←──────────────────────────────────────→ Full LLM Control

[Borg Only]    [LLM Hints]    [LLM Veto]    [LLM Plan]    [LLM Direct]
   0%             20%            50%            80%           100%
```

**Implementation Levels**:

1. **Level 0: Pure Borg** (baseline)
   - Borg runs completely autonomously
   - LLM observes, no intervention
   - Use case: Benchmark borg itself

2. **Level 1: LLM Hints** (20% influence)
   - Borg makes decisions, LLM adjusts borg config parameters
   - Example: "Increase HP_CRITICAL_THRESHOLD from 20% to 30%"
   - Implementation: Config value hot-reload

3. **Level 2: LLM Veto** (50% influence)
   - Borg proposes action, LLM can reject + suggest alternative
   - Example: Borg wants to descend, LLM says "No, rest first"
   - Implementation: Action filter with fallback queue

4. **Level 3: LLM Planning** (80% influence)
   - LLM sets goals, borg executes tactics
   - Example: "Hunt Smeagol on depth 15-20, avoid other combat"
   - Implementation: Goal injection into borg priority system

5. **Level 4: Direct Control** (100% influence)
   - LLM directly commands actions, borg disabled
   - Example: "Move west, attack, quaff healing"
   - Implementation: Action queue bypass

### 3.2 Override Protocol

```cuda
// Device-side borg decision
__device__ BorgAction borg_think(BorgState* state, BorgConfig* config) {
    // Normal borg AI (50+ priority branches)
    BorgAction proposed = borg_full_decision_tree(state, config);

    // Check for LLM override
    if (state->llm_override_active) {
        switch (state->llm_override_type) {
            case OVERRIDE_VETO:
                // LLM rejected this action, try fallback
                if (proposed.type == state->llm_vetoed_action) {
                    return state->llm_suggested_action;
                }
                break;

            case OVERRIDE_GOAL:
                // LLM set goal, borg finds tactics
                BorgGoal llm_goal = state->llm_goal;
                proposed = borg_plan_toward_goal(state, llm_goal);
                break;

            case OVERRIDE_DIRECT:
                // LLM direct control
                return state->llm_action_queue[state->llm_queue_idx++];
        }
    }

    return proposed;
}
```

**Host-side override injection**:
```python
class BorgOverrideManager:
    def __init__(self, gpu_state: CudaMemory):
        self.gpu_state = gpu_state

    async def inject_override(self, instance_id: int, override: LLMOverride):
        # Copy override to GPU memory (interleaved)
        host_data = {
            "override_active": True,
            "override_type": override.type,
            "override_action": override.action,
            "override_duration": override.duration_turns
        }

        # Write to specific instance slot
        offset = instance_id * self.gpu_state.instance_stride
        cuda.memcpy_htod_async(
            self.gpu_state.override_buffer + offset,
            host_data,
            stream=self.gpu_state.stream
        )
```

### 3.3 Override Expression Language

**Natural Language → Structured Override**:

```python
class OverrideParser:
    def parse_llm_command(self, text: str) -> BorgOverride:
        # "Don't go down those stairs"
        if "don't" in text and "stairs" in text:
            return BorgOverride(
                type=OVERRIDE_VETO,
                vetoed_action=BORG_ACTION_DESCEND,
                duration=50  # turns
            )

        # "Use healing NOW"
        if "healing" in text and ("now" in text or "immediately" in text):
            return BorgOverride(
                type=OVERRIDE_DIRECT,
                action=BORG_ACTION_HEAL,
                priority=CRITICAL
            )

        # "Hunt Smeagol but avoid other fights"
        if "hunt" in text and "unique" in text:
            unique_id = self.extract_monster_id(text)
            return BorgOverride(
                type=OVERRIDE_GOAL,
                goal=BorgGoal(
                    type=GOAL_KILL_UNIQUE,
                    target_id=unique_id,
                    constraints=["avoid_other_combat"]
                )
            )
```

**Structured Override API**:
```json
{
    "override_type": "veto",
    "veto": {
        "action": "descend",
        "reason": "Too low HP",
        "suggest": "rest",
        "duration_turns": 100
    }
}

{
    "override_type": "goal",
    "goal": {
        "type": "kill_unique",
        "target": "Smeagol",
        "constraints": ["hp > 80%", "have_teleport"],
        "timeout_turns": 500
    }
}

{
    "override_type": "direct",
    "actions": [
        {"action": "quaff_potion", "item_slot": 3},
        {"action": "read_scroll", "item_slot": 7},
        {"action": "flee", "direction": "any"}
    ],
    "duration_turns": 3
}
```

---

## 4. Cost/Latency Trade-offs

### 4.1 Latency Analysis

**API Call Breakdown** (OpenRouter typical):
```
Network RTT:        20-100ms  (US→API server)
Queue wait:         0-500ms   (load-dependent)
Model compute:      100-2000ms (model size + complexity)
Response streaming: 50-200ms  (token generation)
────────────────────────────────────────────────
Total per call:     170-2800ms
Median:             ~400ms
P95:                ~1200ms
P99:                ~2500ms
```

**GPU Simulation Speed**:
- **DOOM**: 140M ticks/sec (7 μs/tick)
- **Angband**: 4M turns/sec (250 ns/turn)
- **CoreWars**: 40M cycles/sec (25 ns/cycle)

**Latency Impact**:
```
Scenario: Angband, 400ms API latency
- GPU simulates: 400ms × 4M turns/sec = 1.6M turns
- Single instance: 1600 turns (avg game depth 15 = 10K turns)
- Impact: 16% of game frozen per LLM call

Mitigation: Async simulation (continue other instances)
- 10K instances, 1 intervention = 0.016% total throughput loss
```

### 4.2 Cost Analysis

**API Costs** (OpenRouter, Dec 2025):

| Model | Input ($/1M tok) | Output ($/1M tok) | Latency | Use Case |
|-------|------------------|-------------------|---------|----------|
| Gemini 2.5 Flash Lite | $0.04 | $0.15 | 200ms | Fast decisions |
| Grok 4 Fast | $0.15 | $0.60 | 300ms | Balanced |
| DeepSeek V3 | $0.27 | $1.10 | 500ms | Strategic |
| Claude 3.7 Sonnet | $3.00 | $15.00 | 800ms | Research only |

**Token Usage Per Call**:
```python
# State serialization (input)
state_tokens = 200-800  # Game state JSON
context_tokens = 100-300  # Previous decisions
total_input = 300-1100 tokens

# Decision output
decision_tokens = 50-200  # Action + reasoning

total_per_call = 350-1300 tokens
median = ~600 tokens
```

**Cost Per Game** (hybrid protocol, 30-300 calls):

| Game | Calls/Game | Tokens/Game | Cost (Flash Lite) | Cost (Grok Fast) | Cost (Claude) |
|------|------------|-------------|-------------------|------------------|---------------|
| DOOM | 14-24 | 8.4K | $0.002 | $0.006 | $0.15 |
| Angband | 230-280 | 138K | $0.03 | $0.11 | $2.50 |
| CoreWars | 5-8 (tuner) | 3K | $0.0006 | $0.002 | $0.05 |

**Benchmark Scale** (1000 games per model):

| Game | Total Cost (Flash) | Total Cost (Grok) | Total Cost (Claude) | Throughput Loss |
|------|-------------------|-------------------|---------------------|-----------------|
| DOOM | $2 | $6 | $150 | 8% |
| Angband | $30 | $110 | $2,500 | 12% |
| CoreWars | $0.60 | $2 | $50 | 0.5% |

### 4.3 Async Simulation Pattern

**Key Insight**: While instance N waits for LLM, instances 0..N-1 and N+1..10K continue simulating.

```python
class AsyncInterventionManager:
    def __init__(self, gpu_arena, llm_client):
        self.arena = gpu_arena
        self.llm = llm_client
        self.pending_interventions = {}
        self.active_masks = torch.ones(gpu_arena.num_instances, dtype=bool)

    async def simulate_with_intervention(self, num_turns: int):
        for turn in range(num_turns):
            # Check for intervention triggers
            triggers = self.arena.check_triggers()  # GPU → CPU

            for instance_id, trigger in triggers:
                if trigger.should_intervene:
                    # Pause this instance
                    self.active_masks[instance_id] = False

                    # Async LLM call (non-blocking)
                    state = self.arena.get_state(instance_id)
                    task = asyncio.create_task(
                        self.llm.get_decision(state, trigger)
                    )
                    self.pending_interventions[instance_id] = task

            # Simulate ONLY active instances
            active_ids = torch.where(self.active_masks)[0]
            self.arena.step(instance_ids=active_ids)  # GPU continues

            # Check for completed LLM calls
            for instance_id, task in list(self.pending_interventions.items()):
                if task.done():
                    decision = task.result()
                    self.arena.inject_override(instance_id, decision)
                    self.active_masks[instance_id] = True  # Resume
                    del self.pending_interventions[instance_id]

            await asyncio.sleep(0)  # Yield to event loop
```

**Throughput Analysis**:
```
Single-instance freeze time: 400ms (LLM call)
Frozen throughput loss: 400ms × 4M turns/sec = 1.6M turns

Multi-instance async (10K instances):
- 9999 instances continue simulating
- 1 instance frozen
- Throughput loss: 1/10000 = 0.01%

Even with 100 simultaneous interventions:
- 9900 instances active
- Throughput loss: 100/10000 = 1%
```

### 4.4 Batched State Compression

**Problem**: 800 tokens/state × 100 calls/game = 80K tokens input

**Solution**: Differential state updates
```python
class StateDeltaCompressor:
    def __init__(self):
        self.last_states = {}

    def compress(self, instance_id: int, full_state: dict) -> dict:
        if instance_id not in self.last_states:
            # First call: send full state
            self.last_states[instance_id] = full_state
            return full_state

        # Subsequent calls: send only changes
        last = self.last_states[instance_id]
        delta = {}

        for key, value in full_state.items():
            if key not in last or last[key] != value:
                delta[key] = value

        self.last_states[instance_id] = full_state
        return {"_delta": True, "changes": delta}

# Token reduction: 800 → 100-200 (75% savings)
```

---

## 5. Concrete JSON Protocol Specification

### 5.1 Protocol Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM ↔ GPU Protocol v1.0                   │
├─────────────────────────────────────────────────────────────┤
│  Direction    │  Message Type    │  Frequency   │  Size     │
├───────────────┼──────────────────┼──────────────┼───────────┤
│  GPU → LLM    │  STATE_SNAPSHOT  │  On trigger  │  0.5-2KB  │
│  GPU → LLM    │  EVENT_ALERT     │  On event    │  0.3-1KB  │
│  LLM → GPU    │  OVERRIDE_CMD    │  Response    │  0.2-0.8KB│
│  LLM → GPU    │  CONFIG_UPDATE   │  Rare        │  0.1-0.5KB│
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Message Schemas

#### GPU → LLM: STATE_SNAPSHOT
```json
{
  "protocol_version": "1.0",
  "message_type": "state_snapshot",
  "game": "angband",
  "instance_id": 123,
  "turn": 850,
  "wall_time_ms": 12450,

  "trigger": {
    "type": "checkpoint",
    "reason": "periodic_50_turns"
  },

  "player": {
    "hp": 85,
    "max_hp": 120,
    "sp": 12,
    "max_sp": 45,
    "level": 18,
    "xp": 125000,
    "depth": 15,
    "position": {"x": 45, "y": 23},
    "gold": 8500,
    "speed": 10
  },

  "visible": {
    "monsters": [
      {
        "id": 42,
        "name": "Smeagol",
        "unique": true,
        "hp": 150,
        "max_hp": 150,
        "distance": 5,
        "awake": true,
        "danger_score": 350
      },
      {
        "id": 43,
        "name": "Orc warrior",
        "hp": 40,
        "max_hp": 50,
        "distance": 3,
        "danger_score": 80
      }
    ],
    "items": [
      {"type": "potion_healing", "distance": 8},
      {"type": "scroll_teleport", "distance": 12}
    ]
  },

  "inventory": {
    "weapon": {"name": "Long Sword", "damage": "2d5+3", "to_hit": 8},
    "armor": {"name": "Chain Mail", "ac": 30},
    "consumables": {
      "healing_potions": 3,
      "teleport_scrolls": 2,
      "detect_monster_scrolls": 1
    }
  },

  "borg": {
    "autonomous": true,
    "current_plan": "ATTACK",
    "proposed_action": "move_west_and_attack",
    "danger_threshold": 300,
    "hp_critical_pct": 25
  },

  "context": {
    "prev_state_delta": {
      "hp": -15,  // Lost 15 HP since last snapshot
      "depth": 0,  // Same depth
      "monsters_killed": 2
    },
    "session_stats": {
      "turns_elapsed": 850,
      "deepest_depth": 15,
      "total_kills": 87,
      "near_death_events": 2
    }
  }
}
```

#### GPU → LLM: EVENT_ALERT
```json
{
  "protocol_version": "1.0",
  "message_type": "event_alert",
  "game": "angband",
  "instance_id": 123,
  "turn": 850,
  "wall_time_ms": 12450,

  "event": {
    "type": "unique_spotted",
    "severity": "high",
    "monster": {
      "name": "Smeagol",
      "depth": 15,
      "native_depth": 12,
      "distance": 5,
      "current_hp": 150,
      "danger_score": 350
    }
  },

  "player": {
    "hp": 85,
    "max_hp": 120,
    "hp_pct": 71,
    "prepared": false,  // No buffs active
    "can_teleport": true
  },

  "borg": {
    "proposed_action": "ATTACK",
    "confidence": 0.6,
    "reasoning": "Damage output sufficient, HP acceptable"
  },

  "request": {
    "llm_input_needed": true,
    "options": ["approve_borg_plan", "override_with_retreat", "custom_action"],
    "timeout_turns": 5  // Max delay before borg acts anyway
  }
}
```

#### LLM → GPU: OVERRIDE_CMD
```json
{
  "protocol_version": "1.0",
  "message_type": "override_cmd",
  "game": "angband",
  "instance_id": 123,
  "response_to_turn": 850,

  "decision": {
    "override": true,
    "override_type": "direct_action",
    "reason": "Smeagol too dangerous, need preparation",

    "actions": [
      {
        "action": "quaff_potion",
        "item": "blessing",
        "priority": "immediate"
      },
      {
        "action": "read_scroll",
        "item": "teleport",
        "priority": "immediate"
      }
    ],

    "constraints": [
      {
        "type": "action_blacklist",
        "actions": ["descend"],
        "duration_turns": 200,
        "reason": "Avoid deeper dungeon until level 20"
      }
    ],

    "borg_config_adjustments": {
      "danger_flee_threshold": 250,  // More conservative
      "hp_critical_pct": 35  // Retreat earlier
    }
  },

  "metadata": {
    "model": "google/gemini-2.5-flash-lite",
    "latency_ms": 380,
    "tokens_used": {"input": 650, "output": 120},
    "confidence": 0.85
  }
}
```

#### LLM → GPU: CONFIG_UPDATE (TUNER phase)
```json
{
  "protocol_version": "1.0",
  "message_type": "config_update",
  "game": "angband",
  "instance_id": 123,
  "phase": "tuner",

  "config": {
    "borg_strategy": "tank",
    "parameters": {
      "WORSHIPS_HP": true,
      "WORSHIPS_AC": true,
      "PLAYS_RISKY": false,
      "danger_flee_threshold": 200,
      "danger_fight_threshold": 150,
      "hp_critical_pct": 30,
      "hp_rest_threshold": 90,
      "no_deeper_than": 50
    }
  },

  "reasoning": "Conservative tank build for survival testing",

  "metadata": {
    "model": "deepseek/deepseek-v3",
    "based_on_feedback": {
      "prev_run_id": "run_122",
      "prev_strategy": "aggro",
      "prev_outcome": "died_at_depth_8",
      "adjustment": "More defensive parameters"
    }
  }
}
```

### 5.3 Protocol State Machine

```
GPU State:          SIMULATING → WAITING_LLM → APPLYING_OVERRIDE → SIMULATING
                         ↓            ↓                ↓
LLM State:          IDLE → PROCESSING → RESPONDING → IDLE
                         ↑__________________________|
                                (async)

Events:
1. SIMULATING: GPU detects trigger → emit STATE_SNAPSHOT/EVENT_ALERT
2. WAITING_LLM: Pause instance, continue others
3. LLM PROCESSING: Model generates decision (200-2000ms)
4. LLM RESPONDING: Send OVERRIDE_CMD back to GPU
5. APPLYING_OVERRIDE: Update instance state, resume simulation
```

### 5.4 Error Handling

```json
// Timeout (LLM too slow)
{
  "message_type": "error",
  "error_type": "timeout",
  "instance_id": 123,
  "turn": 850,
  "action_taken": "used_borg_fallback",
  "details": "LLM response timeout after 5 turns, borg autonomous decision executed"
}

// Invalid override (action not possible)
{
  "message_type": "error",
  "error_type": "invalid_action",
  "instance_id": 123,
  "requested_action": "cast_spell:fireball",
  "reason": "insufficient_sp",
  "action_taken": "ignored_override",
  "fallback": "borg_decision"
}

// Rate limit hit
{
  "message_type": "error",
  "error_type": "rate_limit",
  "retry_after_ms": 1000,
  "action_taken": "queued_for_retry"
}
```

---

## 6. Cost Estimates

### 6.1 Benchmark Scale Projections

**Assumptions**:
- 10 models to evaluate
- 1000 games per model per game type
- Hybrid protocol (events + checkpoints)

#### DOOM Benchmark
```
Games: 1000 × 10 models = 10,000 games
LLM calls: 10,000 × 18 (avg) = 180,000 calls
Tokens: 180K × 600 (avg) = 108M tokens

Costs:
- Gemini Flash Lite: 108M × $0.10/1M = $10.80
- Grok 4 Fast: 108M × $0.40/1M = $43.20
- DeepSeek V3: 108M × $0.75/1M = $81.00

GPU time: 10K games × 8s = 80,000s = 22 hours
Total cost: $11-81 (depending on model choice)
```

#### Angband Benchmark
```
Games: 1000 × 10 models = 10,000 games
LLM calls: 10,000 × 250 (avg) = 2,500,000 calls
Tokens: 2.5M × 600 (avg) = 1.5B tokens

Costs:
- Gemini Flash Lite: 1.5B × $0.10/1M = $150
- Grok 4 Fast: 1.5B × $0.40/1M = $600
- DeepSeek V3: 1.5B × $0.75/1M = $1,125

GPU time: 10K games × 12s = 120,000s = 33 hours
Total cost: $150-1,125
```

#### CoreWars Benchmark (TUNER only)
```
Warriors: 1000 × 10 models = 10,000 warriors
Generations: 10,000 × 1 (avg) = 10,000 calls
Tokens: 10K × 500 (avg) = 5M tokens

Costs:
- Gemini Flash Lite: 5M × $0.10/1M = $0.50
- Grok 4 Fast: 5M × $0.40/1M = $2.00
- DeepSeek V3: 5M × $0.75/1M = $3.75

Tournament GPU time: 10K warriors × 0.2s = 2,000s = 33 minutes
Total cost: $0.50-3.75 (MUCH cheaper, no runtime intervention)
```

### 6.2 Cost Optimization Strategies

#### Strategy 1: Tiered Model Usage
```python
class TieredLLMStrategy:
    FAST_MODEL = "gemini-flash-lite"  # $0.10/1M tokens
    STRATEGIC_MODEL = "grok-4-fast"   # $0.40/1M tokens

    def route_decision(self, event: Event) -> str:
        if event.severity == "critical" or event.type == "unique_spotted":
            return self.STRATEGIC_MODEL  # 15% of calls
        else:
            return self.FAST_MODEL  # 85% of calls

    # Cost reduction: 85% × $0.10 + 15% × $0.40 = $0.145/1M (27% savings)
```

#### Strategy 2: Delta Compression
```python
# First call: 800 tokens → $0.08
# Subsequent calls: 150 tokens → $0.015
# Average: (800 + 249×150) / 250 = 152 tokens/call
# Savings: 81%
```

#### Strategy 3: Smart Checkpoint Intervals
```python
class AdaptiveCheckpointing:
    def calculate_interval(self, game_state):
        if game_state.danger_level > 300:
            return 10  # Frequent checks in combat
        elif game_state.hp_pct < 50:
            return 20  # Moderate checks when low HP
        else:
            return 100  # Rare checks when safe

    # Reduces checkpoint calls by 60-80% while maintaining quality
```

#### Strategy 4: Batch Parallel Calls
```python
# Sequential: 100 instances × 400ms latency = 40s wall time
# Parallel (batch 50): 2 batches × 400ms = 0.8s wall time
# Throughput: 50× improvement, same cost
```

### 6.3 Free Tier Exploitation

**OpenRouter Free Models** (as of Dec 2025):
- `google/gemini-2.5-flash-lite:free` - 15 RPM limit
- `x-ai/grok-4-fast:free` - 10 RPM limit
- `qwen/qwen3-32b:free` - 20 RPM limit

**Budget Benchmark Strategy**:
```python
class FreeTierManager:
    def __init__(self):
        self.models = [
            ("gemini-flash-lite:free", 15),  # RPM limit
            ("grok-4-fast:free", 10),
            ("qwen-32b:free", 20)
        ]
        self.rate_limiters = {m: RateLimiter(rpm) for m, rpm in self.models}

    async def get_decision(self, state):
        for model, limiter in self.rate_limiters.items():
            if limiter.can_make_request():
                return await self.call_llm(model, state)

        # All rate limited, use borg fallback
        return borg_autonomous_decision(state)

# Total cost: $0
# Constraint: 45 RPM = 2700 calls/hour
# Angband game: 250 calls → 10.8 games/hour
# 1000 games = 93 hours (4 days) per model
```

---

## 7. Recommendations

### 7.1 Optimal Protocol: Hybrid Event + Checkpoint

**Configuration**:

| Game | Checkpoint Interval | Critical Events | Expected Calls/Game | Cost/Game (Flash) |
|------|-------------------|-----------------|---------------------|-------------------|
| DOOM | Every 35 ticks (1s) | HP<30%, ammo=0, boss | 14-24 | $0.002 |
| Angband | Every 50 turns | HP<25%, unique, depth±5 | 230-280 | $0.03 |
| CoreWars | N/A (tuner only) | N/A | 1 | $0.0006 |

**Rationale**:
- Events catch 80% of critical decisions
- Checkpoints catch slow degradation (gradual HP loss, resource depletion)
- Async simulation maintains 90%+ GPU utilization
- Cost scales linearly with game complexity (DOOM cheap, Angband expensive, CoreWars trivial)

### 7.2 Implementation Priorities

**Phase 1: Basic Protocol** (Week 1)
- [ ] Implement STATE_SNAPSHOT message
- [ ] Implement OVERRIDE_CMD parser
- [ ] Add checkpoint-only intervention (simple)
- [ ] Test with single instance, synchronous LLM calls

**Phase 2: Event System** (Week 2)
- [ ] Add event detection to GPU kernels
- [ ] Implement EVENT_ALERT messages
- [ ] Test event-triggered intervention
- [ ] Validate with 100 instances

**Phase 3: Async + Scale** (Week 3)
- [ ] Implement async simulation (pause/resume instances)
- [ ] Add batch LLM calling
- [ ] Optimize state serialization (delta compression)
- [ ] Scale test with 10K instances

**Phase 4: Optimization** (Week 4)
- [ ] Add tiered model routing
- [ ] Implement adaptive checkpoint intervals
- [ ] Add free tier rate limiting
- [ ] Performance tuning (target: <5% throughput loss)

### 7.3 Metrics to Track

**Performance Metrics**:
```python
class BenchmarkMetrics:
    def __init__(self):
        self.gpu_utilization = []  # % time GPU active
        self.llm_latency_p50 = []  # Median API latency
        self.llm_latency_p99 = []  # Tail latency
        self.interventions_per_game = []
        self.tokens_per_game = []
        self.cost_per_game = []

    def target_metrics(self):
        return {
            "gpu_utilization": ">90%",
            "llm_latency_p50": "<400ms",
            "llm_latency_p99": "<1500ms",
            "cost_per_1k_games": "<$50 (Angband), <$5 (DOOM)"
        }
```

**Quality Metrics**:
```python
class DecisionQuality:
    def evaluate(self, game_trace):
        return {
            "llm_overrides_helpful": self.count_helpful_overrides(game_trace),
            "llm_overrides_harmful": self.count_harmful_overrides(game_trace),
            "borg_fallback_rate": self.calculate_fallback_rate(game_trace),
            "game_outcome_improvement": self.compare_vs_pure_borg(game_trace)
        }
```

### 7.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Host (Python)                                │
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │ Intervention │◄────►│     LLM      │◄────►│   OpenRouter │      │
│  │   Manager    │      │   Client     │      │   API Client │      │
│  └──────┬───────┘      └──────────────┘      └──────────────┘      │
│         │                                                             │
│         │ STATE_SNAPSHOT / EVENT_ALERT (async)                      │
│         │ OVERRIDE_CMD / CONFIG_UPDATE                              │
│         ▼                                                             │
│  ┌──────────────────────────────────────────────────────┐           │
│  │            GPU Memory (CUDA)                          │           │
│  │  ┌─────────────────────────────────────────────┐     │           │
│  │  │  Instance State (Interleaved)               │     │           │
│  │  │  [HP][HP][HP]...[X][X][X]...[Depth][Depth]  │     │           │
│  │  └─────────────────────────────────────────────┘     │           │
│  │  ┌─────────────────────────────────────────────┐     │           │
│  │  │  Override Buffer                             │     │           │
│  │  │  [Active][Type][Action][Duration]... (per)  │     │           │
│  │  └─────────────────────────────────────────────┘     │           │
│  │  ┌─────────────────────────────────────────────┐     │           │
│  │  │  Active Instance Mask                        │     │           │
│  │  │  [1][1][0][1][1]... (0 = paused for LLM)    │     │           │
│  │  └─────────────────────────────────────────────┘     │           │
│  └──────────────────────────────────────────────────────┘           │
│         ▲                                                             │
│         │ Launch kernels with active mask                           │
│         │                                                             │
│  ┌──────┴───────────────────────────────────────────────┐           │
│  │              GPU Kernels                              │           │
│  │                                                        │           │
│  │  borg_think_kernel(active_mask) → check_triggers()   │           │
│  │  borg_execute_kernel(active_mask) → apply_overrides()│           │
│  │                                                        │           │
│  └────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘

Data Flow:
1. GPU detects trigger → copy state to host (pinned memory, async)
2. Host sends STATE_SNAPSHOT to LLM (async HTTP)
3. GPU continues simulating active instances (mask excludes paused)
4. LLM responds with OVERRIDE_CMD (200-2000ms later)
5. Host copies override to GPU memory
6. GPU applies override, resumes instance
```

---

## 8. Conclusion

### Key Findings

1. **Hybrid Protocol Optimal**: Events + checkpoints balance cost and control quality
   - DOOM: 14-24 calls/game ($0.002 with Flash Lite)
   - Angband: 230-280 calls/game ($0.03 with Flash Lite)
   - CoreWars: Tuner-only, no runtime intervention

2. **Async Simulation Critical**: Maintains 90%+ GPU utilization despite LLM latency
   - Single-instance freeze: 400ms = 1.6M lost turns
   - 10K-instance async: <1% throughput loss

3. **Cost Scales with Game Complexity**:
   - Simple games (DOOM): $2/1K games
   - Complex games (Angband): $30/1K games
   - Code battles (CoreWars): $0.60/1K games (tuner only)

4. **Action Abstraction Matters**:
   - Atomic: Precise but verbose (19+ actions)
   - Tactical: Balanced, game-specific (8-12 macros)
   - Strategic: Intuitive but complex (4-6 directives)
   - **Recommendation**: Hybrid tactical + strategic

5. **Borg Override Mechanics** (Angband-specific):
   - Veto level (50% influence): Best balance of LLM + borg autonomy
   - Direct control (100% influence): Research only, expensive
   - Hint level (20% influence): Cheap but limited impact

### Production Architecture

```python
class ModelForecastBenchmark:
    """
    Production-ready LLM benchmark with GPU simulation + intervention.
    """

    def __init__(self, game_type: str, num_instances: int = 10000):
        self.game = self._init_game(game_type)  # DOOM/Angband/CoreWars
        self.gpu_arena = GPUArena(game_type, num_instances)
        self.llm_client = AsyncLLMClient(
            model="google/gemini-2.5-flash-lite",
            rate_limiter=RateLimiter(rpm=60)
        )
        self.intervention_mgr = AsyncInterventionManager(
            self.gpu_arena,
            self.llm_client
        )

    async def run_benchmark(self, models: list[str], games_per_model: int):
        results = {}

        for model in models:
            self.llm_client.set_model(model)

            model_results = await asyncio.gather(*[
                self.run_single_game(i)
                for i in range(games_per_model)
            ])

            results[model] = {
                "avg_score": np.mean([r.score for r in model_results]),
                "win_rate": np.mean([r.won for r in model_results]),
                "avg_interventions": np.mean([r.num_interventions for r in model_results]),
                "total_cost": sum(r.cost for r in model_results),
                "gpu_time": sum(r.gpu_time for r in model_results)
            }

        return results

    async def run_single_game(self, game_id: int) -> GameResult:
        # Phase 1: TUNER (pre-game config)
        config = await self.llm_client.generate_config(self.game.type)
        instance_id = self.gpu_arena.init_instance(game_id, config)

        # Phase 2: OBSERVER + INTERVENOR (gameplay with intervention)
        result = await self.intervention_mgr.simulate_with_intervention(
            instance_id=instance_id,
            max_turns=self.game.max_turns,
            protocol=HybridProtocol(
                checkpoint_interval=self.game.checkpoint_interval,
                events=self.game.critical_events
            )
        )

        return result
```

### Next Steps

1. **Prototype Implementation** (2 weeks)
   - Basic JSON protocol
   - Checkpoint-only intervention
   - Single-game validation

2. **Scaling Test** (1 week)
   - 10K instances async simulation
   - Batch LLM calling
   - Cost + throughput measurement

3. **Production Hardening** (2 weeks)
   - Error handling (timeouts, rate limits)
   - State compression (delta updates)
   - Tiered model routing

4. **Full Benchmark** (1 week)
   - 10 models × 1000 games
   - Statistical analysis
   - Publication-ready results

**Total Timeline**: 6 weeks to production benchmark

**Estimated Cost**: $200-1500 depending on game mix and model choice

---

**Document Status**: Complete
**Author**: Claude (Sonnet 4.5)
**Date**: 2025-12-05
**Version**: 1.0
