import json
from openai import OpenAI
from dataclasses import asdict
from .base import ProbeResult
from ..dags.interpreter import DagInterpreter

class DagProbe:
    """
    DAG Benchmark: Runs a full multi-agent workflow.
    """
    def __init__(self, dag_def: dict):
        self.level = 5
        self.name = f"DAG Benchmark: {dag_def.get('workflow_id', 'unknown')}"
        self.dag_def = dag_def
        self.tools = []
        self.prompt = f"Execute DAG: {self.name}" # Dummy prompt for logger

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        # model arg is ignored in favor of DAG definition, 
        # unless we implement override logic later.
        
        interpreter = DagInterpreter(self.dag_def, client)
        result = interpreter.run()
        
        # Serialize result for raw_response
        nodes_dict = {nid: asdict(nres) for nid, nres in result.nodes.items()}
        raw_data = {
            "workflow_id": result.workflow_id,
            "nodes": nodes_dict,
            "total_cost": result.total_cost,
            "success": result.success
        }
        
        return ProbeResult(
            success=result.success,
            tool_called=False, 
            tool_name=None,
            parameters={"total_cost": result.total_cost},
            raw_response=raw_data,
            latency_ms=result.total_latency_ms,
            error=None if result.success else "DAG execution failed"
        )
