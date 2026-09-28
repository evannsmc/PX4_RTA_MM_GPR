# Hardware flight logs (previous version of the code)

These logs, GIFs, PDF frames and notebooks come from the hardware experiments of the paper, flown with the
**previous version of the code**: the single-threaded node on the old `main` branch, now preserved as the git tag
**`archive/main-2026-05-02`** (`git checkout archive/main-2026-05-02`). They are kept exactly as recorded.

* `no_wind_estimation/`, `yes_wind_estimation/`: one `.log` per flight in ROS2Logger's CSV layout (NaN-padded wide
  table, 12 reference / tube rows per control tick), plus the GIFs and PDF frames made from them.
* `plot_hw_logs_*.ipynb`: the notebooks that produced those GIFs / PDFs. They read the old CSV layout and expect the
  code of that tag (e.g. its `TVGPR`), so run them from a checkout of `archive/main-2026-05-02`.

Flights recorded with the current code are HDF5 files in the flight_recorder layout; analyse them with the notebooks
one level up (`../../01_flight_summary.ipynb` ... `../../04_numerical_sim.ipynb`).
