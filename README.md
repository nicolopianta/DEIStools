# DEIStools

DEIStools estimates the (possibly time-varying) multi-frequency impedance of an electrochemical system from voltage/current signals, and visualises the results. It is hardware-agnostic: it works on any suitably sampled signal, not just signals acquired with a specific lab setup.

## Subpackages

- `deistools.processing` — signal processing and impedance/transfer-function estimation from voltage/current signals.
- `deistools.visualise` — plotting helpers for raw signals, spectra, and impedance results.

## Highlights of `deistools.processing`

- `mfa.MultiFrequencyAnalysis` — FFT-EIS (`run_fft_eis`), STFT-EIS, DMFA (`run_dmfa`, and `run_dmfa_fd` with Fermi-Dirac filters whose bandwidths are given in Hz), BLTVA (`run_bltva_eis`, with orders `Np/Nq/Na` and fit diagnostics). `search_freq_indexes` finds each tone's FFT bin without ever picking the DC bin.
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
