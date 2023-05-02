import matplotlib.pyplot as plt
import numpy as np

import os
import json

dir = '/home/buens/Desktop/MA_Evaluation/e4e_encoder_original/webcam_images/results/corrected_original_without_background'
lpips = []
ssim = []
fsim = []

for root, dirs, files in sorted(os.walk(dir)):
    for name in files:
        if name == 'metrics.json':
            id = sorted(dirs)[0]
            with open(os.path.join(dir, id, name)) as f:
                data = json.load(f)
                lpips.append(data['LPIPS'])
                ssim.append(data['SSIM'])
                fsim.append(data['FSIM'])

x = np.arange(1, 101, 1)

# Plotting both the curves simultaneously
plt.plot(x, lpips, color='r', label='LPIPS')
plt.plot(x, ssim, color='g', label='SSIM')
plt.plot(x, fsim, color='b', label='FSIM')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Index")
plt.ylabel("Wert der Metrik")
plt.title("Beleuchtungskorrektur auf Webcam-Aufnahmen")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()


lpips_arr = np.array(lpips)
outliers = np.where(lpips_arr > 0.10)
print(outliers)
