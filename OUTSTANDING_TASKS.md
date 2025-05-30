# Outstanding Tasks

This checklist outlines the remaining work needed to run the Alvarez‑Salvado agents. Work on bilateral sensing is currently on hold.

- [ ] **Environment Setup** – install MATLAB/Octave, add `Code/` to the path, and verify existing tests.
- [ ] **Acquire Required Plume Data** – place the Crimaldi `.hdf5` file in `data/` and obtain a smoke plume `.avi` with known scale and frame rate.
- [x] **Update Configuration Handling** – YAML configs are supported and `run_navigation_cfg.m` respects the `bilateral` flag.
- [x] **Implement Video Looping** – short plume movies now loop automatically when the trial exceeds the video length.
- [ ] **Documentation** – update the README with YAML examples and instructions for looping plume videos.

## Deferred
- [ ] **Add Tests** – verify bilateral config loading and video looping behavior.
- [ ] **Simulation Workflows** – prepare configs for all plume/bilateral combinations and run them to collect results.
- [ ] **Verification** – rerun MATLAB tests to confirm all pass and compare bilateral versus unilateral outputs.
