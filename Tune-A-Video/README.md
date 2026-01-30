# Video Reconstruction via Tune-A-Video

This repository hosts the code and pre-trained weights for reconstructing videos using the **Tune-A-Video** framework. 

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `inference_mindcine.py` | **Main Inference Script.** Run this script to perform video reconstruction. |
| `cond_embeddings.pt` | **[Optimization]** Pre-computed text embeddings for the **Null Text** (empty prompt). |
| `negative.pt` | Pre-computed embeddings for negative prompts (unconditional guidance). |
