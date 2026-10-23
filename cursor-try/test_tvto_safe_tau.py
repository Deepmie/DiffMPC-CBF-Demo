"""Test whether independent Tau snapshots remove the inplace-backward error."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf.base import Tau


def safe_detach(self):
    # Return an independent snapshot instead of detaching/mutating self in place.
    clone = Tau(self._nx, self._nu, self._T, self._tau.detach().clone())
    clone._is_flatten = self._is_flatten
    return clone


Tau.detach = safe_detach

import main

main.main()
