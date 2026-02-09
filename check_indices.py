from panqec.codes import Toric2DCode

def check_stabilizer_index():
    L = 7
    code = Toric2DCode(L)
    stab_map = code.stabilizer_index
    
    # Check if indices are 0...N-1
    indices = sorted(stab_map.values())
    if indices != list(range(len(indices))):
        print("Indices are not contiguous 0...N-1")
    else:
        print("Indices are contiguous 0...N-1")
        
    # Check spatial structure
    # Expected: Row-major order?
    # Or grouped by type (Star vs Plaquette)?
    
    locations = sorted(stab_map.keys(), key=lambda k: stab_map[k])
    
    print("\nFirst 10 locations (Index -> Coord):")
    for i in range(10):
        print(f"{i}: {locations[i]}")

    print("\nLast 10 locations (Index -> Coord):")
    for i in range(len(locations)-10, len(locations)):
        print(f"{i}: {locations[i]}")
        
    # Check if grouped by type
    types = [code.stabilizer_type(loc) for loc in locations]
    print(f"\nTypes first 10: {types[:10]}")
    print(f"Types mid 10: {types[len(types)//2:len(types)//2+10]}")
    
    # Check total length
    print(f"\nTotal Stabilizers: {len(stab_map)}")
    print(f"2 * L * L = {2 * L * L}")
    
    # Check reshape impact
    # If reshape(2, L, L), implies indices 0...L*L are one type, L*L...2*L*L another?
    # Or interleaved?
    
if __name__ == "__main__":
    check_stabilizer_index()
