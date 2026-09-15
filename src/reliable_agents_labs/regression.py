"""Chapter 29: chapter 16 and 21's own evaluation tests each assert one
fixed threshold, 0.75, forever. A real regression can hide above a
fixed floor: a pass rate dropping from 1.00 to 0.80 is worth knowing
about even though 0.80 still clears 0.75. This compares a new run
against the last one actually recorded on disk, in `evals/baselines/`,
not just a fixed number chosen once and never revisited.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASELINES_DIR = Path("evals/baselines")


@dataclass
class Baseline:
    pass_rate: float


def load_baseline(name: str, directory: Path = DEFAULT_BASELINES_DIR) -> Baseline | None:
    """None means no baseline exists yet, a real, distinct case from a
    baseline that exists and was cleared: the very first run of a new
    eval has nothing to regress against.
    """
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Baseline(pass_rate=data["pass_rate"])


def save_baseline(name: str, pass_rate: float, directory: Path = DEFAULT_BASELINES_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"pass_rate": pass_rate}, indent=2))


def check_for_regression(
    name: str,
    current_pass_rate: float,
    tolerance: float = 0.1,
    directory: Path = DEFAULT_BASELINES_DIR,
) -> str | None:
    """`None` means no regression, either because the current run held
    up or because no baseline exists yet to regress against, a first
    run is a beginning, not a failure. A real message names both real
    numbers, never a bare boolean a caller has to go dig the cause of.
    """
    baseline = load_baseline(name, directory)
    if baseline is None:
        return None
    if current_pass_rate < baseline.pass_rate - tolerance:
        return (
            f"{name}: pass rate dropped from {baseline.pass_rate:.2f} to "
            f"{current_pass_rate:.2f}, more than the {tolerance:.2f} tolerance"
        )
    return None
