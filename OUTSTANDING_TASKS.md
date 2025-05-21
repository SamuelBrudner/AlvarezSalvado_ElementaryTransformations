# Outstanding Tasks

This checklist outlines the remaining work needed to run the Alvarez‑Salvado agents with and without bilateral sensing using both the Crimaldi plume and a custom smoke plume movie.

- [ ] **Environment Setup** – install MATLAB/Octave, add `Code/` to the path, and verify existing tests.
- [ ] **Acquire Required Plume Data** – place the Crimaldi `.hdf5` file in `data/` and obtain a smoke plume `.avi` with known scale and frame rate.
- [ ] **Update Configuration Handling** – create YAML configs for bilateral and unilateral options and update `run_navigation_cfg.m` to respect a `bilateral` flag.
- [ ] **Implement Video Looping** – modify the `video` case in `navigation_model_vec.m` to repeat short plume movies.
- [ ] **Add Tests** – verify bilateral config loading and video looping behavior.
- [ ] **Simulation Workflows** – prepare configs for all plume/bilateral combinations and run them to collect results.
- [ ] **Documentation** – update the README with YAML examples and instructions for looping plume videos.
- [ ] **Verification** – rerun MATLAB tests to confirm all pass and compare bilateral versus unilateral outputs.

