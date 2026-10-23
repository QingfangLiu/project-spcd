# SPCD Task Overview  
**Sensory Preconditioning and Devaluation**

## Summary  
The SPCD task is a multi-day, fMRI-compatible behavioral paradigm designed to study outcome devaluation and sensory preconditioning. It includes multiple phases such as cue-cue learning, cue-odor conditioning, value updating through devaluation, and recognition memory testing. Participants complete tasks both inside and outside the MRI scanner.

Participants are asked to **fast for 2–4 hours before arrival** to ensure consistent hunger state for the devaluation manipulation. The full study is completed over **three sessions (Day 0, Day 1, and Day 2)**, typically scheduled within a **2-week window**, although exceptions may occur in practice.

---

## Subject Info File

`subject_info.csv` is a truncated subject-info file copied from the full
subject-info file in the data folder. It keeps only the columns needed by this
repo-level analysis context.

The `pairing` column records cue-category counterbalancing:

- `c`: conditioned cues are characters
- `p`: conditioned cues are places

With the current cue coding, conditioned cues are even `cue_index` values
`2, 4, 6, 8, 10, 12`, and preconditioned cues are odd `cue_index` values
`1, 3, 5, 7, 9, 11`.

---

## Day-by-Day Procedure

### Day 0 – Baseline Rating & Odor Selection
- **Image Rating Task**  
  Participants rate character and place images, which are later used as visual cues in the experiment.
- **Initial Odor Rating & Selection**  
  Participants rate the pleasantness and intensity of several odors. Two odors—one savory and one sweet—are selected for each participant based on these ratings.
- **Follow-Up Odor Rating**  
  Participants re-rate the same odors to ensure consistency and confirm odor selection.
- **Odor Discrimination Task**  
  Participants judge whether two presented odors are the same or different.

---

### Day 1 – Pre-conditioning & Localizer (fMRI)
- **Localizer Task (in scanner)**  
  Participants undergo fMRI scans to measure neural responses to individual images and odors, before any associative learning has occurred.  
  - odor runs 
  - character runs  
  - place runs  
  - anatomical scan
- **Pre-conditioning Training (outside scanner)**  
  Participants learn associations between characters and places.  
  

---

### Day 2 – Conditioning, Probe, Devaluation
- **Pre-conditioning Refresher (outside scanner)**  
  A brief session to reinforce previously learned character-place associations.  
- **Odor Prediction Task (in scanner)**  
  Participants predict which odor follows a given conditioned cue.  
- **Pre-meal Probe (in scanner)**  
  Tests participants’ responses to **pre-conditioned and conditioned cues**, without any odor delivery.  
- **Meal & Devaluation Manipulation**  
  - Pre-meal odor ratings and questions.
  - Participants consume a meal flavored with one of the two previously conditioned odors.
  - Post-meal odor ratings and questions.
- **Post-meal Probe (in scanner)**  
  Identical to the pre-meal probe but administered after devaluation.  
- **Recognition Memory Task (outside scanner)**  
  Participants judge whether specific character-place pairs were seen during pre-conditioning.  
- **Post-experiment Interview**  
  A short debriefing or follow-up assessment to gather subjective feedback or check for awareness.

---


## Task Timing and Run Structure Details

This section summarizes the trial timing, run structure, and duration for each task used in the SPCD paradigm.

### Day 1 (MRI Localizer + Pre-conditioning)

#### **Odor Scan (in scanner)**
- **Trial**: 2s sniff cue → 2s odor → 9–13s ITI  
- **Structure**: 3 odors × 10 repetitions = 30 trials per run  
- **Duration**: ~7.5 min per run  
- **Runs**: 2 (320 volumes/run)  

#### **Character Image Scan**
- **Trial**: 2s image cue → 2s image → 5–8s ITI  
- **Structure**: 8 images (6 task + 2 filler) × 4 repetitions = 32 trials  
- **Duration**: ~5.6 min per run  
- **Runs**: 4 (244 volumes/run)  

#### **Place Image Scan**
- Same as above  

#### **Pre-conditioning Training (outside scanner)**
- **Trial**: 3s first cue → 0.3s gap → 3s second cue (choice) → 3–8s ITI  
- **Structure**: 6 cue pairs × 2 alternatives × L/R = 24 trials  
- **Duration**: ~5 min per run  
- **Runs**: 3–5; task may end after the 3rd or 4th run if **≥90% accuracy** is achieved.

---

### Day 2 (Refresher + Conditioning + Devaluation + Memory)

#### **Pre-conditioning Refresher (outside scanner)**
- Same structure as Day 1  
- **1–3 runs** (max 3); task may end after 1 run if **≥90% accuracy** is achieved.

#### **Odor Prediction (in scanner)**
- **Trial (programmed timing)**: 2s cue → 3s prediction → 2s delay → 2s odor release → 4–10s ITI
- **Odor arrival**: The effective delay from the end of prediction to odor arrival at the nose is approximately **4s**: the **2s programmed delay** plus approximately **2s transit through the delivery tubing** after odor release. Figure 1 and its caption use this effective delay.
- **Structure**: 6 cues × 6 orderings = 36 trials  
- **Duration**: ~10 min per run  
- **Runs**: 3 (404 volumes/run)  
- **Notes**: Each cue appears 18 times across all runs

#### **Pre-meal Probe (in scanner)**
- **Trial**: 3s cue1 → 4–10s gap → 3s cue2 → 2s decision → 4–10s ITI  
- **Structure**: 6 pairs × 2 sets × L/R = 24 trials  
- **Duration**: ~9 min per run  
- **Runs**: 3 (372 volumes/run)  
- **Notes**: Each cue appears 12 times across all runs

#### Meal Time
- ~15 min meal

#### **Post-meal Probe (in scanner)**
- Identical to pre-meal probe  

#### **Recognition Memory (outside scanner)**
- **Trial**: 3s cue1 → 0.3s gap → 3s cue2 → 3–8s ITI  
- **Structure**: 6 cue pairs × old/foil × 3 repetitions = 36 trials  
- **Duration**: ~7 min  
- **Runs**: 1  

---

### Odor–Food Mapping

The table below lists the specific foods that were used to devalue each odor.

| Odor           | Food(s)                                   |
|----------------|-------------------------------------------|
| Potato Chip    | Potato chips                              |
| Garlic         | Minced garlic jars, Naan                  |
| Cheddar        | Mac n cheese, Goldfish crackers, Cheeze-Its |
| Pizza          | Frozen pizzas, Tomato sauce jars          |
| Tuna           | Canned tuna, Crackers                     |
| Strawberry     | Strawberry strudels                       |
| Chocolate      | Clif bars                                 |
| Apple cinnamon | Nutrigrain bars, Apple crisps             |
| Peach          | Fruit cups, Peach strudels                |
| Vanilla        | Vanilla cookies                           |



---

## Notes on data issue handling

> This only documents issues that **may affect later analysis**. Some abnormal data collection events are **not noted here** if they have already been well handled and do **not impact** downstream analyses.
> Most issue-related corrections (e.g., scan truncation, trigger adjustments) are handled in the [`Do_5_process_labchart.m`](./scripts_MRI/preproc_organize_imaging/Do_5_process_labchart.m) script.

- **Sub-35 – Day 2, Odor Prediction Run 3**  
  The scan did not complete all expected volumes. Processed the LabChart trigger data to use only the partial scan. Also corrected a minor mismatch with the number of motion parameters.

- **Sub-36 – Day 2, Pre-Meal Probe Run 3**  
  The scan was interrupted mid-run, resulting in a short initial recording followed by a second, longer segment. Both were excluded from analysis to avoid alignment issues, since the behavioral task was not paused.

- **Sub-14 – Day 2, Meal**  
  - Subject was initially assigned to the savory–devaluation group, but due to a lack of pizza in storage, was fed the sweet meal instead.  
  - No code changes were made to preserve consistent labeling across tasks.

- **Sub-33 – Day 2, Post-Meal Probe Task, Run 3**  
  - Breathing signals were missing, likely due to a connection issue.  
  - No correction was applied; breathing signals were extracted as usual.
