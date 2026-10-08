import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _file_dialog():
    """tkinter is only needed by the interactive loaders below; importing it lazily lets
    this module be used on machines/environments without a display or Tk."""
    from tkinter import Tk, filedialog
    root = Tk()  # pointing root to Tk() to use it as Tk() in program.
    root.withdraw()  # Hides small tkinter window.
    root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite of selection.
    return filedialog


def load_impedance_set(text : str = 'Select npy impedance file'):
    # Open dialog window to select pkl file containing data
    filedialog = _file_dialog()
    # Returns file path as str
    filename = filedialog.askopenfilename(title = text)
    directory = os.path.dirname(filename)
    # Import impedance data
    Z = np.load(filename)
    return Z, directory

def load_impedance_set_with_error(text : str = 'Select npy impedance file'):
    # Open dialog window to select pkl file containing data
    filedialog = _file_dialog()
    filename = filedialog.askopenfilename(title = text)
    directory = os.path.dirname(filename)
    # Import impedance data
    Z = np.load(filename)
    Zerr = np.load(directory+"/error_low_freq.npy")
    metadata_dir = os.path.dirname(os.path.dirname(directory))
    with open(metadata_dir + "/metadata_deis_exp.json", 'r') as file:
        frequencies = json.load(file)["Frequencies remaining (Hz)"]
    return Z, Zerr, frequencies, directory

def load_voltage_current(text : str = "Select directory with voltage and current npy files"):
    # Open dialog window to select pkl file containing data
    filedialog = _file_dialog()
    directory = filedialog.askdirectory(title = text)
    # Import impedance data
    voltage = np.load(directory + '/voltage.npy')
    current = np.load(directory + '/current.npy')
    return voltage, current, directory

def load_pyeclab_data(text: str = "Select csv file"):
    import pandas as pd
    filedialog = _file_dialog()
    file_path = filedialog.askopenfilename(title = text)
    return pd.read_csv(file_path, sep = '\t')


# ---------------------------------------------------------------------------#
# Non-interactive loading of a whole experiment folder (no tkinter).

def find_technique_folders(experiment_folder):
    """
    `pico_aquisition/cycle_<loop>_sequence_<technique>/` folders of an experiment, in
    run order. The name match is strict, so a leftover `cycle_*_manual_stop` copy from
    an older version is not loaded on top of its original.
    """
    pico = Path(experiment_folder) / "pico_aquisition"
    found = []
    if pico.is_dir():
        for d in pico.iterdir():
            m = re.fullmatch(r"cycle_(\d+)_sequence_(\d+)", d.name)
            if m and d.is_dir():
                found.append(((int(m[1]), int(m[2])), d))
    return [d for _, d in sorted(found)]


@dataclass
class ExperimentData:
    folder: Path
    time: np.ndarray = None                  # s, from measurement_data.txt
    potential: np.ndarray = None             # V
    current: np.ndarray = None               # A
    frequencies: np.ndarray = None           # Hz, the rows of `impedance`
    impedance: np.ndarray = None             # (n_freq, n_blocks) complex, empty blocks dropped
    decimated_voltage: np.ndarray = None     # V, all techniques concatenated in run order
    decimated_current: np.ndarray = None     # A
    technique_folders: list = field(default_factory=list)
    # How many blocks / decimated samples each technique folder contributed to `impedance` / `decimated_*` (in run
    # order, only folders that had the file): the boundaries of the pieces a series is made of, e.g. for
    # deistools.processing.smoothing.segment_bounds(...).
    block_counts: list = field(default_factory=list)
    sample_counts: list = field(default_factory=list)
    notes: list = field(default_factory=list)  # one human-readable line per thing loaded/missing/failed


def load_experiment(experiment_folder) -> ExperimentData:
    """
    Load what a DEIS experiment folder holds: `measurement_data.txt` (potentiostat
    time/Ewe/I), `metadata_deis_exp.json` (frequency list), and the online results --
    impedance, decimated voltage and current -- of every technique folder, concatenated
    in run order. Experiment folders from before the per-technique layout kept the three
    arrays directly in the experiment folder (`impedance.npy`, `decimated_voltage.npy`,
    `decimated_current.npy`); those are read when no technique folder exists.
    Nothing raises for a missing or unreadable part: it is left as None and explained
    in `.notes`.
    """
    folder = Path(experiment_folder)
    data = ExperimentData(folder=folder)
    notes = data.notes

    measurement_path = folder / "measurement_data.txt"
    if measurement_path.exists():
        try:
            try:
                import pandas as pd
                table = pd.read_csv(measurement_path, delimiter="\t").to_numpy(dtype=float)
            except ImportError:
                table = np.loadtxt(measurement_path, skiprows=1)
            data.time, data.potential, data.current = table[:, 0], table[:, 1], table[:, 2]
            notes.append(f"measurement_data.txt: {data.time.size:,} rows")
        except Exception as exc:
            notes.append(f"measurement_data.txt: FAILED ({type(exc).__name__}: {exc})")
    else:
        notes.append("measurement_data.txt: not found")

    metadata_path = folder / "metadata_deis_exp.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            # "STFFT-EIS frequencies" is what impedance.npy's rows actually match (see
            # the acquisition software's metadata writer); fall back to the full multisine list, which is the
            # same thing unless a subset was excluded from the online analysis.
            freqs = meta.get("STFFT-EIS frequencies (Hz)") or meta.get("Frequencies multisine (Hz)")
            if freqs:
                data.frequencies = np.asarray(freqs, dtype=float)
                notes.append(f"metadata_deis_exp.json: {data.frequencies.size} frequencies")
            else:
                notes.append("metadata_deis_exp.json: no frequency list found in it")
        except Exception as exc:
            notes.append(f"metadata_deis_exp.json: FAILED ({type(exc).__name__}: {exc})")
    else:
        notes.append("metadata_deis_exp.json: not found")

    data.technique_folders = find_technique_folders(folder)
    if data.technique_folders:
        sources = [(d / "impedance.npy", d / "voltage.npy", d / "current.npy") for d in data.technique_folders]
        where = ", ".join(d.name for d in data.technique_folders)
    else:
        sources = [(folder / "impedance.npy", folder / "decimated_voltage.npy", folder / "decimated_current.npy")]
        where = "experiment folder (older layout)"

    try:
        imps = [np.load(i) for i, _, _ in sources if i.exists()]
        if imps:
            kept = [piece[:, np.any(np.abs(piece) > 0, axis=0)] for piece in imps]    # empty blocks dropped
            data.block_counts = [piece.shape[1] for piece in kept]
            data.impedance = np.hstack(kept)
            notes.append(f"impedance: {data.impedance.shape[1]} block(s) x {data.impedance.shape[0]} freq(s) [{where}]")
            if data.frequencies is not None and data.frequencies.size != data.impedance.shape[0]:
                notes.append(
                    f"  WARNING: frequency count ({data.frequencies.size}) doesn't match "
                    f"the impedance array's rows ({data.impedance.shape[0]})."
                )
        else:
            notes.append("impedance: not found")
    except Exception as exc:
        notes.append(f"impedance: FAILED ({type(exc).__name__}: {exc})")

    try:
        pairs = [(np.load(v), np.load(c)) for _, v, c in sources if v.exists() and c.exists()]
        if pairs:
            data.decimated_voltage = np.concatenate([v for v, _ in pairs])
            data.decimated_current = np.concatenate([c for _, c in pairs])
            data.sample_counts = [v.size for v, _ in pairs]
            notes.append(f"decimated voltage/current: {data.decimated_voltage.size:,} samples")
        else:
            notes.append("decimated voltage/current: not found")
    except Exception as exc:
        notes.append(f"decimated voltage/current: FAILED ({type(exc).__name__}: {exc})")
    return data


if __name__ == '__main__':
    load_impedance_set()
    load_voltage_current()
