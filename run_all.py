"""Run all analyses and produce all figures of the paper.

    python run_all.py            # analyses + figures
    python run_all.py --figures  # figures only (from the stored results)
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

ANALYSES = ["s1_suspension.py", "s2_aero_bem.py", "s3_statics.py", "s3b_connection.py", "s4_vessel.py",
            "s4b_vessel_sensitivity.py", "s5_dynamics.py", "s5b_checks.py", "s5c_operability.py",
            "s6_control.py", "s6b_tuggers.py", "s7_verify_6dof.py"]
FIGURES = ["make_figures.py", "fig1_schematic.py", "graphical_abstract.py"]


def run(script):
    t0 = time.time()
    print(f"--> {script}", flush=True)
    res = subprocess.run([sys.executable, script], cwd=SRC)
    if res.returncode != 0:
        sys.exit(f"{script} failed with exit code {res.returncode}")
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    scripts = FIGURES if "--figures" in sys.argv else ANALYSES + FIGURES
    for s in scripts:
        run(s)
    print("All results are in results/ and all figures in figures/.")
