import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button
from tkinter import Tk, filedialog

# Disable tkinter root window
Tk().withdraw()

# Select folder with images
folder_path = filedialog.askdirectory(title="Select Folder with Images")

# Supported image extensions
extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Get all image paths
image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(extensions)]

# Storage for results
results = []

# State variables
selected_point = None

def process_image(img_path):
    global selected_point
    selected_point = None

    # Load image using OpenCV
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(img_rgb)
    def onclick(event):
        global selected_point
        if event.inaxes == ax:  # Only register clicks on the image axes
            if event.xdata is not None and event.ydata is not None:
                selected_point = (int(event.xdata), int(event.ydata))
                print(f"Selected point: {selected_point}")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    def confirm(event):
        if selected_point is not None:
            x, y = selected_point
            rgb = img_rgb[y, x]
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[y, x]
            print(f"RGB: {rgb}, HSV: {hsv}")
            results.append({
                "image_name": os.path.basename(img_path),
                "x": x,
                "y": y,
                "R": int(rgb[0]),
                "G": int(rgb[1]),
                "B": int(rgb[2]),
                "H": int(hsv[0]),
                "S": int(hsv[1]),
                "V": int(hsv[2])
            })
            fig.canvas.mpl_disconnect(cid)  # disconnect the click event
            plt.close()

    ax_confirm = plt.axes([0.4, 0.05, 0.2, 0.075])
    b_confirm = Button(ax_confirm, 'Confirm')
    b_confirm.on_clicked(confirm)

    plt.show()

# Process each image
for img_path in image_paths:
    print(f"Processing: {img_path}")
    process_image(img_path)

# Save results
df = pd.DataFrame(results)
csv_path = "colorsExtraction/color_points.csv"
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")
