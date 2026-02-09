import stim
import numpy as np

def check_stim_obs_padding():
    # Circuit 1: Define obs 0 and 1
    c1 = stim.Circuit()
    c1.append("M", [0, 1])
    c1.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)
    c1.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    
    det1, obs1 = c1.compile_detector_sampler().sample(shots=1, separate_observables=True)
    print(f"Obs 0,1 shape: {obs1.shape}")
    
    # Circuit 2: Define obs 2 and 3 ONLY
    c2 = stim.Circuit()
    c2.append("M", [0, 1])
    c2.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 2)
    c2.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 3)
    
    det2, obs2 = c2.compile_detector_sampler().sample(shots=1, separate_observables=True)
    print(f"Obs 2,3 shape: {obs2.shape}")
    print(f"Obs 2,3 content (should be random since M is random? No, initialized |0> so 0): {obs2}")
    
    # Check if obs2 has 4 cols or 2 or 4 (dense)?
    
if __name__ == "__main__":
    check_stim_obs_padding()
