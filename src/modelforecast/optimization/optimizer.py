import json
import os
from pathlib import Path
from openai import OpenAI
from modelforecast.dags.interpreter import DagInterpreter
from typing import Dict, Any

class DagOptimizer:
    def __init__(self, dag_def: Dict[str, Any], client: OpenAI):
        self.dag_def = dag_def
        self.client = client

    def optimize_loop(self, iterations: int = 3):
        print(f"Optimizing DAG: {self.dag_def.get('workflow_id', 'unknown')}")
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            
            # Run DAG
            interpreter = DagInterpreter(self.dag_def, self.client)
            result = interpreter.run()
            
            print(f"Result: Success={result.success}, Cost=${result.total_cost:.6f}, Latency={result.total_latency_ms}ms")
            
            if result.success:
                print("DAG succeeded. Checking for cost optimizations...")
            else:
                print("DAG failed. Attempting to fix...")
                
            # Teacher Step
            suggestion = self.get_teacher_suggestion(result)
            if suggestion:
                self.apply_suggestion(suggestion)
            else:
                print("No suggestions from Teacher.")
                break
                
    def get_teacher_suggestion(self, result):
        # Construct prompt for Teacher
        context = f"""
        DAG Workflow: {self.dag_def.get('workflow_id', 'unknown')}
        Nodes: {len(self.dag_def.get('nodes', []))}
        Success: {result.success}
        Total Cost: ${result.total_cost:.6f}
        
        Node Results:
        """
        for node_id, res in result.nodes.items():
            context += f"- {node_id}: Status={res.status}, Cost=${res.cost:.6f}, Error={res.error}\n"
            if res.status == "failed":
                context += f"  Output/Error details: {res.output or res.error}\n"

        prompt = f"""
        You are an AI Optimizer (Teacher).
        Analyze the execution of this DAG.
        Goal: Improve reliability and reduce cost (target 15% reduction) while maintaining quality.
        
        Current Execution Context:
        {context}
        
        Current DAG Definition (Nodes):
        {json.dumps(self.dag_def.get('nodes', []), indent=2)}
        
        Suggest a specific change to ONE node's prompt to improve the result.
        If successful, try to shorten prompts to save tokens.
        If failed, try to make prompts more explicit to fix errors.
        
        Return JSON with keys: "node_id", "new_prompt", "reasoning".
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash-lite-preview-09-2025",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Teacher failed: {e}")
            return None

    def apply_suggestion(self, suggestion):
        node_id = suggestion.get("node_id")
        new_prompt = suggestion.get("new_prompt")
        
        if not node_id or not new_prompt:
            return

        print(f"Applying suggestion for {node_id}: {suggestion.get('reasoning')}")
        
        for node in self.dag_def.get("nodes", []):
            if node["id"] == node_id:
                old_len = len(node.get("prompt", ""))
                new_len = len(new_prompt)
                node["prompt"] = new_prompt
                print(f"Updated prompt (Length: {old_len} -> {new_len})")
                return
