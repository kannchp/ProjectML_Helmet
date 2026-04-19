import pandas as pd
import matplotlib.pyplot as plt
import os

# โหลด csv
df = pd.read_csv("./basemodel/basemodel_results .csv")
# สร้าง folder เก็บกราฟ
os.makedirs("plots", exist_ok=True)

# metric ที่อยาก plot
metrics = [
    "train/box_loss",
    "train/cls_loss",
    "val/box_loss",
    "val/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

epoch = df["epoch"]

for metric in metrics:
    if metric in df.columns:
        plt.figure()

        plt.plot(epoch, df[metric], marker='o')

        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)

        plt.grid(True)

        # save รูป
        filename = metric.replace("/", "_").replace("(", "").replace(")", "")
        plt.savefig(f"plots/{filename}.png")

        plt.close()

print("Saved plots in ./plots/")