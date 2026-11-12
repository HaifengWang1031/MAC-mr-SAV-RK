"""Historical CLI/import adapter to postprocess.snapshots; never computes a flow."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess.snapshots import main as main, report as report


if __name__ == '__main__':
    main()
