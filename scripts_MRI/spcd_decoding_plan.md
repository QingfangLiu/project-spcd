# 🧠 SPCD fMRI Decoding Plan: Next Steps

---

## 🗂 Cue Mapping Structure

| **Cue Index** | **Cue Set** | **Pre-conditioned Cue** | **Conditioned Cue** | **Associated Outcome (Odor)** |
|---------------|-------------|-------------------------|---------------------|--------------------------------|
| 1             | Set 1       | A1<sub>1</sub>          |                     | odor_1                         |
| 2             | Set 1       |                         | B1<sub>1</sub>      | odor_1                         |
| 3             | Set 1       | A2<sub>1</sub>          |                     | odor_2                         |
| 4             | Set 1       |                         | B2<sub>1</sub>      | odor_2                         |
| 5             | Set 1       | A3<sub>1</sub>          |                     | clean air                      |
| 6             | Set 1       |                         | B3<sub>1</sub>      | clean air                      |
| 7             | Set 2       | A1<sub>2</sub>          |                     | odor_1                         |
| 8             | Set 2       |                         | B1<sub>2</sub>      | odor_1                         |
| 9             | Set 2       | A2<sub>2</sub>          |                     | odor_2                         |
| 10            | Set 2       |                         | B2<sub>2</sub>      | odor_2                         |
| 11            | Set 2       | A3<sub>2</sub>          |                     | clean air                      |
| 12            | Set 2       |                         | B3<sub>2</sub>      | clean air                      |


- A cues = **Pre-conditioned** (learned through A→B pairing)
- B cues = **Conditioned** (directly paired with odors)
- Odor outcomes: **odor_1**, **odor_2** (odor that was devalued via meal), or **clean air**

---

## 🎯 Goal 1: Pre-conditioned → Conditioned Cue Reactivation

### **Research Question**
Do **pre-conditioned cues (Day 2, Probe)** evoke neural patterns similar to **conditioned cues (Day 1)**?

### **Design**
**Train on:**  
- Conditioned cues (B1–B6) from Day 1 scan  
- 6-way classification (cue identity) using trial-wise GLM beta maps

**Test on:**  
- Pre-conditioned cues (A1–A6) from Day 2 Probe  
- Each A cue labeled by its associated B cue (e.g., A1 → B1 → class 0)

### **Interpretation**
Above-chance accuracy (chance = 1/6) suggests associative reactivation of conditioned cues through **model-based inference**.

---

## 🎯 Goal 2: Odor Identity Reactivation

### **Research Questions**
1. Do **pre-conditioned cues** evoke reactivation of their **associated odor identity**?  
2. Do **conditioned cues** evoke their **directly associated odor identity**?

### **Design**

**Train on:**  
- Day 1 odor scan (odor_1, odor_2, clean air)

#### Test 1: On Pre-Conditioned Cues (A1–A6)
- Label each A cue based on the odor linked via its associated B cue (e.g., A1 → B1 → odor_1)

#### Test 2: On Conditioned Cues (B1–B6)
- Label each B cue directly by its odor association (e.g., B1 → odor_1)

### **Interpretation**
- Test 1: Reactivation of odor identity via **two-step associative inference**
- Test 2: Reactivation of **directly learned odor associations**

---

## 🎯 Goal 3: Cross-Set & Cross-Cue-Type Odor Identity Decoding

### **Research Question**
Is **odor identity** represented in a form that generalizes across:
- Different **cue types** (pre-conditioned vs conditioned), and  
- Different **cue sets** (Set 1 vs Set 2)?

### **Approach 1: Cross-Set Decoding (within cue type)**
- **Train:** On **Set 1** cues of one cue type  
  - Pre-conditioned: A1<sub>1</sub>, A2<sub>1</sub>, A3<sub>1</sub>  
  - Conditioned: B1<sub>1</sub>, B2<sub>1</sub>, B3<sub>1</sub>  
  (labeled by odor identity)
- **Test:** On **Set 2** cues of the *same* cue type  
  - Pre-conditioned: A1<sub>2</sub>, A2<sub>2</sub>, A3<sub>2</sub>  
  - Conditioned: B1<sub>2</sub>, B2<sub>2</sub>, B3<sub>2</sub>
👉 Tests generalization across visually distinct cues of the **same type**

### **Approach 2: Cross-Cue-Type + Cross-Set Decoding**
- **Train:** On **Set 1** cues of one cue type  
  - Example: Pre-conditioned (A1<sub>1</sub>, A2<sub>1</sub>, A3<sub>1</sub>)
- **Test:** On **Set 2** cues of the *other* cue type  
  - Example: Conditioned (B1<sub>2</sub>, B2<sub>2</sub>, B3<sub>2</sub>)
👉 Tests generalization across both **cue type** and **cue set**

⚠️ Controls for cue–cue association confound by using **non-overlapping sets**

---

## 🎯 Goal 4: Conditioned Cue → Pre-conditioned Cue Reactivation

### **Research Question**
Do **conditioned cues (B1–B6, Day 2 Odor Prediction)** evoke neural patterns similar to their **associated pre-conditioned cues (A1–A6, Day 1)**?

### **Design**

**Train on:**  
- Pre-conditioned cues (A1–A6) from Day 1 scan  
- Each trial labeled by A cue identity (0–5)

**Test on:**  
- Conditioned cues (B1–B6) from Day 2 odor prediction task  
- Each B cue labeled by its associated A cue (e.g., B1 → A1 → class 0)

### **Interpretation**
Above-chance decoding suggests **reactivation of pre-conditioned cue representations** during viewing of associated conditioned cues, implying **bidirectional associative structure**.

---

## 🎯 Goal 5: Representational Similarity to Odor Templates

### **Research Question**
Does the similarity between cue responses and odor templates reflect **devaluation-based value updating**?

### **Design**
- Build **odor_1** and **odor_2** templates from Day 1 odor scan
- For each cue (A or B), compute similarity to odor templates:
  - **Before the meal**
  - **After the meal**

### **Interpretation**
- Similarity to **devalued odor template** should **decrease** post-meal  

