import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modelforecast.runner import ProbeRunner

def main():
    # Target model
    model = "deepseek/deepseek-v3.2-exp"
    
    # Output dir
    output_dir = Path(__file__).parent.parent / "results" / "deepseek_exp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running benchmarks for {model}...")
    
    try:
        runner = ProbeRunner(
            output_dir=output_dir,
            models=[model],
            contributor="jw409-gemini-agent"
        )
        
        # Run 5 trials per level to be quick but representative
        runner.run_all(trials=5, max_level=4)
        
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
