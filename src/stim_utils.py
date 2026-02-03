import stim
import numpy as np
from panqec.codes import StabilizerCode

def get_stabilizer_connectivity(code: StabilizerCode):
    """
    Analyze stabilizers to determine connections and scheduling.
    Returns a list of steps, where each step is a list of (ancilla, data) pairs for CNOTs.
    """
    # Map stabilizer location to index
    stab_map = code.stabilizer_index
    qubit_map = code.qubit_index
    
    # Directions: (dx, dy)
    # We want to group CNOTs by direction to enable parallel execution (layers).
    # Standard Toric Code directions from get_stabilizer: (-1,0), (1,0), (0,-1), (0,1)
    # i.e. West, East, South, North.
    # Order: N, E, S, W is often good. Or Z: N,E,S,W / X: N,E,S,W.
    
    directions = [
        (0, 1),   # North
        (1, 0),   # East
        (0, -1),  # South
        (-1, 0),  # West
    ]
    
    schedule = {d: [] for d in directions}
    
    # Iterate over all stabilizers
    for loc, s_idx in stab_map.items():
        # Get operator dict: {qubit_loc: 'X' or 'Z'}
        op = code.get_stabilizer(loc)
        
        for q_loc, pauli in op.items():
            # Determine direction d = (qx-sx, qy-sy) wrapped
            # Wait, code.size = (Lx, Ly)
            Lx, Ly = code.size
            dx = (q_loc[0] - loc[0])
            dy = (q_loc[1] - loc[1])
            
            # Unwrap periodic boundary
            # Allowed coords are 0..2L.
            # Delta should be roughly +/- 1.
            # If |dx| > 1, it wrapped.
            if dx > 1: dx -= 2*Lx
            if dx < -1: dx += 2*Lx
            if dy > 1: dy -= 2*Ly
            if dy < -1: dy += 2*Ly
            
            d = (dx, dy)
            if d not in schedule:
                # Should not happen for standard toric code
                continue
                
            q_idx = qubit_map[q_loc]
            schedule[d].append((s_idx, q_idx, pauli))
            
    return schedule, directions

def generate_stim_circuit(code: StabilizerCode, rounds: int, p: float, q: float, basis: str = "Z") -> stim.Circuit:
    c = stim.Circuit()
    
    qubit_map = code.qubit_index
    num_data = code.n
    num_stab = len(code.stabilizer_index)
    
    # Precompute types
    types = [""] * num_stab
    for loc, s_idx in code.stabilizer_index.items():
        types[s_idx] = code.stabilizer_type(loc)
    
    # Indices: 0..num_data-1 (Data), num_data..num_data+num_stab-1 (Ancilla)
    
    # Reset Data
    # Reset Data
    if basis == "Z":
        c.append("R", range(num_data))
    else:
        c.append("RX", range(num_data))
        
    if p > 0:
        c.append("DEPOLARIZE1", range(num_data), p)
    
    schedule, directions = get_stabilizer_connectivity(code)
    
    # Helper to get global indices
    def anc(s_idx): return num_data + s_idx
    def data(q_idx): return q_idx

    # Measurement loops
    for round_i in range(rounds):
        # 1. Init Ancillas
        z_stabs = [i for i, t in enumerate(types) if t == 'vertex']
        x_stabs = [i for i, t in enumerate(types) if t == 'face']
        
        # Reset all ancillas
        c.append("R", [anc(i) for i in range(num_stab)])
        # H on X-stabilizer ancillas to measure X
        c.append("H", [anc(i) for i in x_stabs])
        
        # 2. Apply CNOTs in layers
        for d in directions:
            targets = []
            ops = schedule[d]
            for (s_idx, q_idx, pauli) in ops:
                if pauli == 'Z':
                    targets.append(data(q_idx))
                    targets.append(anc(s_idx))
                elif pauli == 'X':
                    targets.append(anc(s_idx))
                    targets.append(data(q_idx))
            
            if targets:
                c.append("CX", targets)
                if p > 0:
                    c.append("DEPOLARIZE2", targets, p)
        
        # 3. Measure Ancillas
        c.append("H", [anc(i) for i in x_stabs])
        if q > 0:
            c.append("X_ERROR", [anc(i) for i in range(num_stab)], q)  
        c.append("M", [anc(i) for i in range(num_stab)])
        
        # 4. Detectors
        # |0> is superposition of X. X-check outcomes are random +1/-1.
        # Wait. If data is initialized to |0>, Z-stabilizers are deterministic (0). 
        # X-stabilizers are random (50/50).
        # So we can only define detectors for X-stabilizers after the *second* round?
        # Or we initialize in a logical state?
        # Standard practice: "Fault tolerant initialization" or just accept random first round for X.
        # Usually for QEC training: We only care about VALID detectors (detecting errors).
        # Detectors for Z-stabs: M[t] ^ M[t-1].
        # Detectors for X-stabs: M[t] ^ M[t-1].
        # Is X-stab M[0] meaningful? 
        # If we start with random X-outcomes, M[0] is random. M[1] will match M[0] if no errors.
        # So Detector is ALWAYS M[t] ^ M[t-1].
        # For t=0: Compare to "reset"? No, just record M[0]. Detector requires M[0] ^ M[-1].
        # So we usually define detectors starting from round 1.
        
        # Let's declare detectors for t > 0
                
        if round_i > 0:
            for i in range(num_stab):
                current_rec = stim.target_rec(-num_stab + i)
                prev_rec = stim.target_rec(-2*num_stab + i)
                c.append("DETECTOR", [current_rec, prev_rec], [0.0]*3)
        else:
            for i in range(num_stab):
                current_rec = stim.target_rec(-num_stab + i)
                if types[i] == 'vertex':
                    c.append("DETECTOR", [current_rec], [0.0]*3)
                else:
                    c.append("DETECTOR", [current_rec, current_rec], [0.0]*3)
 

    # 5. Define Observables
    # We want to track Logical X and Logical Z errors.
    # panqec code.logicals_x is a list of operator dicts (one for each logical qubit).
    # Since we use 2D Toric Code, usually k=2 (2 logical qubits).
    # 5. Define Observables
    # If basis="Z": Track Logical Z (sensitive to X errors). Logical X is random.
    # If basis="X": Track Logical X (sensitive to Z errors). Logical Z is random.
    
    # Logical X operators
    log_x = code.get_logicals_x()
    for i, op in enumerate(log_x):
        if basis == "X":
            targets = []
            for loc, pauli in op.items():
                targets.append(stim.target_x(qubit_map[loc]))
            c.append("OBSERVABLE_INCLUDE", targets, i) # Index i
        
    # Logical Z operators
    log_z = code.get_logicals_z()
    offset = len(log_x)
    for i, op in enumerate(log_z):
        if basis == "Z":
            targets = []
            for loc, pauli in op.items():
                targets.append(stim.target_z(qubit_map[loc]))
            c.append("OBSERVABLE_INCLUDE", targets, offset + i) 
                
    return c
