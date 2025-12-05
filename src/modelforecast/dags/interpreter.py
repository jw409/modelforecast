import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from openai import OpenAI
from modelforecast.metrics.cost import CostTracker

@dataclass
class NodeResult:
    node_id: str
    output: str
    tool_calls: List[Dict[str, Any]]
    latency_ms: int
    cost: float
    status: str # "success", "failed"
    error: Optional[str] = None

@dataclass
class DagResult:
    workflow_id: str
    nodes: Dict[str, NodeResult]
    total_cost: float
    total_latency_ms: int
    success: bool

class DagInterpreter:
    def __init__(self, dag_def: Dict[str, Any], client: OpenAI):
        self.dag_def = dag_def
        self.client = client
        self.cost_tracker = CostTracker()
        self.results: Dict[str, NodeResult] = {}

    def run(self) -> DagResult:
        start_total = time.time()
        total_cost = 0.0
        
        nodes = self.dag_def.get("nodes", [])
        pending_nodes = {n["id"]: n for n in nodes}
        completed_nodes = set()
        
        while pending_nodes:
            progress = False
            # Simple loop to find runnable nodes
            # Sort by id for deterministic order if multiple are ready
            for node_id in sorted(pending_nodes.keys()):
                node = pending_nodes[node_id]
                deps = node.get("depends_on", [])
                
                if all(d in completed_nodes for d in deps):
                    # Ready to run
                    result = self.execute_node(node)
                    self.results[node_id] = result
                    completed_nodes.add(node_id)
                    del pending_nodes[node_id]
                    progress = True
                    total_cost += result.cost
                    
                    if result.status == "failed":
                        return DagResult(
                            workflow_id=self.dag_def.get("workflow_id", "unknown"),
                            nodes=self.results,
                            total_cost=total_cost,
                            total_latency_ms=int((time.time() - start_total) * 1000),
                            success=False
                        )
                    break # Restart loop to respect dependencies properly
            
            if not progress and pending_nodes:
                # Cycle detected or missing dep
                break
                
        return DagResult(
            workflow_id=self.dag_def.get("workflow_id", "unknown"),
            nodes=self.results,
            total_cost=total_cost,
            total_latency_ms=int((time.time() - start_total) * 1000),
            success=len(pending_nodes) == 0
        )

    def execute_node(self, node: Dict[str, Any]) -> NodeResult:
        start_time = time.time()
        prompt = node.get("prompt", "")
        # Respect the model in the DAG, but allow override if needed (not implemented here)
        model = node.get("model", "google/gemini-2.5-flash-lite-preview-09-2025:free")
        
        # Context injection: Append previous outputs
        context_str = "\n\n--- CONTEXT FROM PREVIOUS STEPS ---\n"
        has_context = False
        for dep_id in node.get("depends_on", []):
            if dep_id in self.results:
                context_str += f"\n[Output from {dep_id}]:\n{self.results[dep_id].output}\n"
                has_context = True
        
        full_prompt = prompt
        if has_context:
            full_prompt += context_str
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
            )
            
            latency = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            output = choice.message.content or ""
            
            usage = response.usage
            cost = 0.0
            if usage:
                cost = self.cost_tracker.calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
                
            return NodeResult(
                node_id=node["id"],
                output=output,
                tool_calls=[], 
                latency_ms=latency,
                cost=cost,
                status="success"
            )
            
        except Exception as e:
            return NodeResult(
                node_id=node["id"],
                output="",
                tool_calls=[],
                latency_ms=int((time.time() - start_time) * 1000),
                cost=0.0,
                status="failed",
                error=str(e)
            )
