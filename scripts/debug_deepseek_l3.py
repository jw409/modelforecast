import sys
import os
import json
from pathlib import Path
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modelforecast.runner import ProbeRunner
from modelforecast.probes.level3_multiturn import Level3MultiTurnProbe
from openai import OpenAI

def debug_l3():
    console = Console()
    model = "deepseek/deepseek-v3.2-exp"
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    probe = Level3MultiTurnProbe()
    
    console.print(f"[bold]Debugging {model} on Level 3[/bold]")
    
    # Turn 1
    console.print("\n[yellow]Turn 1: 'Find files related to authentication and read the most relevant one.'[/yellow]")
    probe.turn1_prompt = "Find files related to authentication and read the most relevant one."
    turn1 = probe._execute_turn1(model, client)
    
    if not turn1.success:
        console.print(f"[red]Turn 1 Failed: {turn1.error}[/red]")
        console.print(f"Response: {turn1.raw_response}")
        return

    console.print(f"[green]Turn 1 Success: Called {turn1.tool_name}[/green]")
    console.print(f"Params: {turn1.parameters}")
    
    # Turn 2
    console.print("\n[yellow]Turn 2: Injecting results...[/yellow]")
    turn2 = probe._execute_turn2(model, client, turn1.tool_name, turn1.parameters)
    
    if turn2.success:
        console.print(f"[green]Turn 2 Success: Called {turn2.tool_name}[/green]")
    else:
        console.print(f"[red]Turn 2 Failed: {turn2.error}[/red]")
        
        # Inspect what it actually did
        if turn2.raw_response:
            choice = turn2.raw_response['choices'][0]
            msg = choice['message']
            content = msg.get('content')
            tool_calls = msg.get('tool_calls')
            
            console.print(f"\n[bold]Actual Response Analysis:[/bold]")
            console.print(f"Content: {content}")
            console.print(f"Tool Calls: {tool_calls}")
            
            if content and not tool_calls:
                console.print("\n[blue]Hypothesis: Model stopped to chat instead of acting.[/blue]")

if __name__ == "__main__":
    debug_l3()
