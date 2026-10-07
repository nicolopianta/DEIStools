# DEIStools

DEIStools estimates the (possibly time-varying) multi-frequency impedance of an electrochemical system from voltage/current signals, and visualises the results. It is hardware-agnostic: it works on any suitably sampled signal, not just signals acquired with a specific lab setup.

## Subpackages

### `deistools.processing`

Signal processing and impedance/transfer-function estimation from voltage/current signals.

- **`MultiFrequencyAnalysis`** — main analysis class. Computes FFT, runs FFT-EIS, STFT-EIS, DMFA (Dynamic Multi-Frequency Analysis), and BLTVA (Band-Limited Time-Varying Analysis) on multi-frequency voltage/current data.
- **`FermiDiracFilter`** — symmetric Fermi-Dirac shaped digital bandpass filter used by DMFA.
- **`BLTVAResult`** — result container for the BLTVA estimator, including time-varying transfer functions, drift estimates, and diagnostics.

### `deistools.simulation`

Synthetic signal generation for testing and benchmarking estimation algorithms. A `simulate` engine generates multi-frequency current excitation and the corresponding voltage response from a user-supplied impedance model, with optional measurement noise. Three ready-made circuit models are included:

- **`simulate_r_rc`** — stationary R + RC circuit (series resistance plus parallel RC element).
- **`simulate_r_rc_transistor`** — non-stationary R + (R+T)||C circuit where a transistor-controlled resistance produces a time-varying impedance driven by a gate voltage ramp.
- **`simulate_legendre`** — Legendre polynomial oracle model where the transfer function varies in time as a Legendre expansion, matching the BLTVA signal model for exact-recovery testing.

### `deistools.visualise`

Plotting helpers for raw signals, spectra, and impedance results.

- **`plot_technique`** — two-panel time-series plot of voltage and current.
- **`inspect_spectrum` / `inspect_spectrum_phase`** — interactive magnitude/phase spectrum inspector with range slider.
- **`visualise_peaks` / `visualise_peaks_phase`** — interactive peak inspector at excited frequencies.
- **`plot_impedance_set`** — interactive Nyquist plot with time-index slider.
- **`plot_impedance_set_with_error`** — Nyquist plot plus error-vs-frequency panel with time-index slider.

## Examples

The [`examples/`](examples/) directory contains runnable scripts demonstrating simulation and visualisation workflows:

- [`stationary_r_rc.py`](examples/stationary_r_rc.py) — simulate a stationary R+RC circuit with multi-frequency excitation and visualise the time-domain signals and Nyquist plot.
- [`nonstationary_r_rc_transistor.py`](examples/nonstationary_r_rc_transistor.py) — simulate a non-stationary circuit with transistor-controlled impedance drift and visualise how the Nyquist plot evolves over time.

## Highlights of `deistools.processing`

- `mfa.MultiFrequencyAnalysis` — FFT-EIS (`run_fft_eis`), STFT-EIS, DMFA (`run_dmfa`, and `run_dmfa_fd` with Fermi-Dirac filters whose bandwidths are given in Hz), BLTVA (`run_bltva`, `visualize_bltva_fit`). `search_freq_indexes` finds each tone's FFT bin without ever picking the DC bin.
- `detrending` — `remove_baseline`/`redo_baseline` (per-window line) and `detrend(...)`: end-point line, polynomial or smooth-spline trend removal for long records before the FFT (the trend is returned so it can be added back to the DC result). The spline needs `scipy` (`pip install deistools[spline]`).
- `dmfa_pipeline` — `run_dmfa_on_record(...)`: detrend, DMFA and DC extraction on a whole decimated record in one call; `zero_order_hold_resample` to put coarser results on the DMFA time axis.
- `data_loader` — `load_experiment(folder)` / `find_technique_folders(folder)` read the `pico_aquisition/cycle_<loop>_sequence_<technique>/` experiment folders written by the DEIS acquisition software, without any GUI toolkit.
- `spectrum` — `amplitude_spectrum`, `log_binned_max` for plotting long spectra.

## Tests

```
python -m unittest discover -s tests -t .
```

## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. It is no longer part of it: deistools only works on arrays and files, and is hardware-agnostic. The last combined snapshot remains available at this repo's `v0.1.0` tag.
