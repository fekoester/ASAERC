__version__ = "0.2.0"

from .data import load_data_files_whole
from .readouts import AttentionReadout2DPDE
from .reservoir import PDE2DReservoir, build_pde_reservoir_compiled
from .simulation import simulate_pde_for_file
from .train import PDETrainer

__all__ = [
    "load_data_files_whole",
    "simulate_pde_for_file",
    "PDE2DReservoir",
    "build_pde_reservoir_compiled",
    "AttentionReadout2DPDE",
    "PDETrainer",
]
