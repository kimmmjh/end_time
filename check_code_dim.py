import hydra
import numpy as np
from hydra.utils import instantiate
from panqec.codes import StabilizerCode
from src._data_generator import DataGenerator

@hydra.main(config_path="config", config_name="2d", version_base="1.2")
def check_code(args):
    L = 5
    print(f"Instantiating code with L={L}")
    code: StabilizerCode = instantiate(args.default.code, L)
    
    print(f"Code Type: {type(code)}")
    print(f"n (qubits): {code.n}")
    print(f"k (logical qubits): {code.k}")
    print(f"d (distance): {code.d}")
    
    print(f"Stabilizer Matrix Shape: {code.stabilizer_matrix.shape}")
    print(f"Logicals X shape: {code.logicals_x.shape}")
    print(f"Logicals Z shape: {code.logicals_z.shape}")
    
    gen = DataGenerator(code=code, error_rate=0.1, batch_size=1, circuit_noise=True)
    print(f"Generator Logicals Shape (stacked): {gen.logicals.shape}")
    
    if gen.logicals.shape[0] != 4:
        print("WARNING: Expected 4 logical operators (X1, Z1, X2, Z2) for k=2. Found different amount.")
    else:
        print("SUCCESS: 4 Logical Operators found (consistent with 2D Toric Code).")

if __name__ == "__main__":
    check_code()
