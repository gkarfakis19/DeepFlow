import sys
from pathlib import Path

# This is magic file that pytest loads. We need it unfortunately to hook the validation_helpers module into pytest.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

pytest_plugins = ["validation_scripts.validation_helpers"]
