# test_environment.py

import numpy as np
from src.environment import SatelliteEnv
from pettingzoo.test import parallel_api_test

# File: test_environment.py
# Thay thế hàm run_tests cũ bằng hàm này

def run_tests():
    """
    Function to run all tests for the custom environment.
    """
    print("=====================================================")
    print("        TESTING SATELLITE ENVIRONMENT")
    print("=====================================================")

    # --- Test 1: PettingZoo API Compliance ---
    print("\n[INFO] Running PettingZoo API compliance test...")
    try:
        env_for_api_test = SatelliteEnv()
        parallel_api_test(env_for_api_test, num_cycles=1000)
        print("[SUCCESS] PettingZoo API compliance test passed!")
    except Exception as e:
        print(f"[ERROR] API compliance test failed: {e}")
        return

    # --- Test 2: Manual Run-through ---
    print("\n[INFO] Running manual run-through test...")
    try:
        env = SatelliteEnv(n_agents=3, n_tasks=5)
        observations, infos = env.reset()
        
        agent_0_id = env.possible_agents[0]

        print(f"  - Number of agents: {len(env.possible_agents)}")
        print(f"  - First agent ID: {agent_0_id}")
        # Sửa cách lấy sample space
        print(f"  - Observation space shape: {env.observation_space(agent_0_id).shape}")
        print(f"  - Action space size: {env.action_space(agent_0_id).n}")

        print(f"  - Initial observation shape for {agent_0_id}: {observations[agent_0_id].shape}")

        max_steps = 5
        for step in range(max_steps):
            if not env.agents:
                break
                
            # Sửa cách lấy sample action
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            print(f"\n  --- Step {step+1}/{max_steps} ---")
            print(f"    Actions: {actions}")
            print(f"    Rewards: {rewards}")
            print(f"    Terminations: {terminations}")
            print(f"    Truncations: {truncations}")
            
            if all(terminations.values()) or all(truncations.values()):
                print("  Episode finished.")

        env.close()
        print("\n[SUCCESS] Manual run-through test completed.")
    except Exception as e:
        print(f"[ERROR] Manual run-through test failed: {e}")
        import traceback
        traceback.print_exc() # In ra lỗi chi tiết hơn

    print("\n=====================================================")
    print("              ALL TESTS COMPLETED")
    print("=====================================================")


if __name__ == "__main__":
    run_tests()
    
