"""Fail closed on incompatible training/scoring environments."""
import platform
import sys
import joblib
import numpy
import pandas
import scipy
import sklearn
from .io import require

def check_runtime(expected=None):
    require(sys.version_info[:2] == (3, 12), "Use the pinned Python 3.12 container.")
    require(platform.machine() == "s390x", "Classifier execution requires native s390x.")
    require(sklearn.__version__ == "1.9.0", "Use pinned scikit-learn 1.9.0.")
    current = {
        "python": platform.python_version(), "architecture": platform.machine(),
        "interpreter": sys.executable, "sklearn": sklearn.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "pandas": pandas.__version__, "joblib": joblib.__version__,
    }
    if expected is not None:
        for key in ("python", "architecture", "sklearn", "numpy", "scipy", "pandas", "joblib"):
            require(current[key] == expected[key], "Saved model environment mismatch: " + key)
    return current
