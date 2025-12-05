import torch
import time

def prototype_gpu_angband():
    print("Initializing GPU Angband Prototype...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU for structural demo.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using {torch.cuda.get_device_name(0)}")

    # --- PARAMETERS ---
    BATCH_SIZE = 1024  # 1024 Parallel Universes
    HEIGHT = 66
    WIDTH = 198
    MAX_MONSTERS = 64
    
    print(f"Allocating memory for {BATCH_SIZE} instances...")
    
    # --- 1. THE DUNGEON TENSOR (SoA) ---
    # Replaces 'struct square'
    # Channels:
    # 0: Feature ID (Wall, Floor, Door)
    # 1: Monster Index (0 if empty, 1-64 otherwise)
    # 2: Object Index (0 if empty)
    # 3: Lighting Info (Bitmask)
    dungeon_tensor = torch.zeros(
        (BATCH_SIZE, 4, HEIGHT, WIDTH), 
        dtype=torch.uint8, 
        device=device
    )
    
    # --- 2. THE MONSTER POOL (ECS) ---
    # Replaces 'm_list' linked list
    # Attributes: [Active, RaceID, Y, X, HP, Energy]
    monster_pool = torch.zeros(
        (BATCH_SIZE, MAX_MONSTERS, 6), 
        dtype=torch.int16, 
        device=device
    )
    
    # --- 3. THE PLAYER TENSOR ---
    # Attributes: [Y, X, HP, MaxHP, Energy]
    players = torch.zeros(
        (BATCH_SIZE, 5), 
        dtype=torch.int16, 
        device=device
    )
    
    # Initialize dummy data (Vectorized initialization)
    print("Initializing dummy state...")
    
    # All players at (10, 10)
    players[:, 0] = 10 # Y
    players[:, 1] = 10 # X
    players[:, 2] = 100 # HP
    
    # Create walls (Feature ID 1) at borders
    # This operation happens for ALL 1024 instances simultaneously
    dungeon_tensor[:, 0, 0, :] = 1 # Top wall
    dungeon_tensor[:, 0, HEIGHT-1, :] = 1 # Bottom wall
    dungeon_tensor[:, 0, :, 0] = 1 # Left wall
    dungeon_tensor[:, 0, :, WIDTH-1] = 1 # Right wall
    
    # Spawn a monster in every instance at (20, 20)
    monster_pool[:, 0, 0] = 1 # Active
    monster_pool[:, 0, 1] = 101 # Race ID: Filthy Street Urchin
    monster_pool[:, 0, 2] = 20 # Y
    monster_pool[:, 0, 3] = 20 # X
    monster_pool[:, 0, 4] = 10 # HP
    
    # Update Grid to reflect monster presence (Kernel Logic Simulation)
    # In a real kernel, this would be atomic or handled by the move kernel
    # Here we just use advanced indexing
    batch_indices = torch.arange(BATCH_SIZE, device=device)
    dungeon_tensor[batch_indices, 1, 20, 20] = 1 # Set Monster Index 1 at (20,20)
    
    print("Memory Allocated. Running Micro-Kernel Benchmark...")
    
    # --- BENCHMARK: "MOVE PLAYER" KERNEL ---
    # Simple logic: Try to move East. If wall, stay.
    
    start_time = time.time()
    NUM_STEPS = 1000
    
    for step in range(NUM_STEPS):
        # 1. Calculate candidate position (East)
        current_y = players[:, 0]
        current_x = players[:, 1]
        target_x = current_x + 1
        
        # 2. Check collision (Vectorized Lookup)
        # We need to grab the Feature ID at (current_y, target_x) for each batch
        # gather is slow, advanced indexing is better
        target_features = dungeon_tensor[batch_indices, 0, current_y.long(), target_x.long()]
        
        # 3. Predicate: Is it a wall (ID 1)?
        # (1 is wall, 0 is floor in this dummy setup)
        # If target_features == 1 (Wall), mask = 0 (Don't move), else 1 (Move)
        can_move = (target_features != 1)
        
        # 4. Update Position (Masked Update)
        players[:, 1] = torch.where(can_move, target_x, current_x)
        
        # (Optional) Synchronize to measure true kernel time
        if step % 100 == 0:
             torch.cuda.synchronize() if torch.cuda.is_available() else None
             
    end_time = time.time()
    total_ops = BATCH_SIZE * NUM_STEPS
    duration = end_time - start_time
    sps = total_ops / duration
    
    print(f"--- RESULTS ---")
    print(f"Steps Per Second: {sps:,.0f}")
    print(f"Total Duration: {duration:.4f}s")
    print(f"Final Player X (Instance 0): {players[0, 1].item()}")
    print(f"Final Player X (Instance 500): {players[500, 1].item()}")
    
    # Validation
    # Wall is at WIDTH-1 (197). Player started at 10.
    # Should hit wall at 197-1 = 196
    
if __name__ == "__main__":
    prototype_gpu_angband()
