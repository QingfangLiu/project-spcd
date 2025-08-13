## Cross-cue decoding (Approach 2)

This analysis extends the **cross-set decoding** procedure to test **cross-cue generalization**:

- **Goal:** Train a classifier on one cue type (A or B) in one set, then test it on the *other* cue type in the *other* set.
  - Example fold: Train on **Set 1 / Cue A**, test on **Set 2 / Cue B**.
  - Second fold: Train on **Set 1 / Cue B**, test on **Set 2 / Cue A**.
- This evaluates whether learned neural patterns for one cue type generalize to the *other* cue type in the opposite set.

---

### Key changes from cross-set decoding

1. **Data selection**
   - Cross-set decoding: Training and testing used the *same cue type* across sets (e.g., Cue B in Set 1 → Cue B in Set 2).
   - Cross-cue decoding: Training and testing use *different* cue types (A ↔ B) across sets.

2. **Normalization**
   - Cross-set decoding: Run-wise z-scoring **within each set**.
   - Cross-cue decoding: Run-wise z-scoring **within each set *and cue type*** to avoid cue-type information leakage.

3. **Modes**
   - **`crosscue_set2set`**: Pooled across all runs in each set (train all runs of Set 1 / Cue A → test all runs of Set 2 / Cue B, and vice versa).
   - **`crosscue_strict`**: Run-matched folds (for each run, train on other runs’ Set 1 / Cue A → test that run’s Set 2 / Cue B, and vice versa).

4. **Output naming**
   ```
   <base_dir>/searchlight_maps/probe_crosscue/
     sub-XX_session-premeal_mode-<crosscue_set2set|crosscue_strict>_cue-AB_classes-<2|3>_radius-8mm.nii.gz
   ```

5. **Swarm generation**
   - New script: `make_searchlight_swarm_crosscue.py` generates job files for both modes and class counts.
   - Supports the same run-on-Biowulf workflow as cross-set decoding.

