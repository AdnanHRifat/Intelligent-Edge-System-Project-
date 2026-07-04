# EdgeShield Project

This project contains an edge AI demo for detecting target assets such as laptops, phones, and people using Ultralytics YOLO.

## Run locally

1. Install dependencies:
   ```bash
   pip install streamlit ultralytics opencv-python-headless psutil
   ```
2. Start the app:
   ```bash
   streamlit run Final_demo.py
   ```

## Notes

- Large model weights and generated output folders are ignored by Git.
- The app can use a local OpenVINO model if present; otherwise it falls back to a PyTorch model.
