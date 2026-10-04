from .cost import Cost, QuadCost
from .dynamic import Dynamic, LinDynamic
from .base import MetricFunction, iLQRMetricFunction, Tau
from .diffmpc import DiffMPC

def main() -> None:
    print("Hello from diffmpc-cbf!")
