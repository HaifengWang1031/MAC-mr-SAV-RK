"""Serial postprocessing of explicit MPI run records."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cli import analyze_main
if __name__=='__main__':analyze_main()
