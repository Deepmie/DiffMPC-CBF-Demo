from .cost import Cost, QuadCost
from .dynamic import Dynamic, LinDynamic
from .metricsfunc import MetricFunction, iLQRMetricFunction
from .base import Tau
from .diffmpc import DiffMPC

def main() -> None:
    print("Hello from diffmpc-cbf!")
