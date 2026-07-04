from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time

# 1. Models to compare for Final 6-Model Matrix
models_to_test = {
    "Nano (Baseline)": "yolov8n.pt",
    "Small (Native)": "yolov8s.pt",
    "Medium (PyTorch)": "yolov8m.pt",
    "Next-Gen (Small)": "yolo11s.pt",  # Instantly downloads YOLO11-Small
    "Large (Stress Test)": "yolov8l.pt",
    "Medium (OpenVINO)": "yolov8m_openvino_model"
}

results_data = []

print("--- STARTING MASTER 6-MODEL VALIDATION ---")

for name, path in models_to_test.items():
    print(f"\n[STEP] Testing {name}...")
    
    try:
        # Load the model
        model = YOLO(path, task='detect')
        
        # Format a safe folder name for the runs directory
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # Run validation with plots=True to generate Confusion Matrix, F1 Curves, etc.
        # This guarantees you get the visual validation images for your report!
        metrics = model.val(data='coco8.yaml', imgsz=640, plots=True, name=f"final_eval_{safe_name}")
        
        # Extract Latency Metrics from Ultralytics internal speed tracker
        speed_data = getattr(metrics, 'speed', {})
        inference_time = speed_data.get('inference', 0.0)
        
        # Extract Accuracy Metrics
        map_50 = metrics.box.map50
        precision = metrics.box.mp
        recall = metrics.box.mr
        
        # Calculate F1 Score mathematically
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results_data.append({
            "Model": name,
            "Latency (ms)": inference_time,
            "mAP@50": map_50,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        })
        print(f"DONE: {name} metrics and plots saved to runs/detect/final_eval_{safe_name}")
        
    except Exception as e:
        print(f"[ERROR] Failed to validate {name}: {e}")

# 2. CONVERT DATA TO DATAFRAME
if not results_data:
    print("No data collected. Exiting.")
    exit()
    
df = pd.DataFrame(results_data)
print("\n--- FINAL 6-MODEL DATA COLLECTED ---")
print(df[['Model', 'Latency (ms)', 'mAP@50', 'F1-Score', 'Precision']])

# 3. VISUALIZATION - BAR CHART
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

ax = sns.barplot(x="Model", y="Latency (ms)", data=df, palette="viridis")

# Add data labels onto the bar chart
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    label_text = f"mAP: {df['mAP@50'].iloc[i]:.3f}\nF1: {df['F1-Score'].iloc[i]:.3f}"
    ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02), 
            label_text, ha="center", va="bottom", 
            fontweight='bold', fontsize=9, color='black')

plt.title("Final Benchmarking: Speed vs. Accuracy (F1 & mAP)", fontsize=16, pad=30)
plt.ylabel("Inference Latency (ms) - Lower is Better", fontsize=12)
plt.xlabel("Model Architecture", fontsize=12)
plt.ylim(0, df['Latency (ms)'].max() * 1.3)
plt.xticks(rotation=15) # Rotate x-labels slightly so they don't overlap

plt.tight_layout()
plt.savefig("final_6model_comparison.png", dpi=300)
print("\n--- SUCCESS: 'final_6model_comparison.png' saved! ---")

# 4. VISUALIZATION - HEATMAP
# Prepare data for a heatmap comparison
heatmap_data = df.set_index('Model')[['mAP@50', 'F1-Score', 'Precision', 'Recall']]

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Score'})

plt.title("Model Accuracy Heatmap: 6-Model Comparative Analysis", fontsize=15)
plt.tight_layout()
plt.savefig("final_6model_heatmap.png", dpi=300)
print("--- SUCCESS: 'final_6model_heatmap.png' saved! ---")

print("\n🎯 ALL DONE! Go to your 'runs/detect/' folder to find the Confusion Matrices and F1 curves for the final report!")