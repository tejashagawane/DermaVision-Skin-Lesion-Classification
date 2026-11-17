# DermaVision — Streamlit UX refresh

## Files
- `app.py` — redesigned Streamlit interface
- `requirements.txt` — Python dependencies
- `models/` — place the trained checkpoint here

## Required model
Copy your trained file to:

`models/ham10000_resnet50_best.pth`

The checkpoint must contain `model_state_dict` and preferably `label_to_idx` / `idx_to_label`, and must match the seven-class ResNet50 architecture and preprocessing used in training.

## Run on Windows CMD

```cmd
cd /d C:\skin_lesion_streamlit
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you extract this ZIP into a different folder, change the `cd` command to that folder.

## Notes
This is a research/educational prototype, not a clinical diagnostic device. Softmax scores are not calibrated clinical probabilities.
