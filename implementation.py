from tkinter import Tk, filedialog
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import joblib
import pandas as pd

model = joblib.load('elasticnet_pipeline.pkl')

selected_point = None
ph_value = None

Tk().withdraw()  # Hide main Tk window
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

if not image_path:
    print("No image selected.")
    exit()

img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)
ax.imshow(img_rgb)
ax.set_title("Click a point and enter pH")

def onclick(event):
    global selected_point
    if event.inaxes == ax:
        if event.xdata is not None and event.ydata is not None:
            selected_point = (int(event.xdata), int(event.ydata))
            print(f"Selected point: {selected_point}")

def on_submit_ph(text):
    global ph_value
    try:
        ph_value = float(text)
        print(f"pH value set to: {ph_value}")
    except ValueError:
        print("Invalid pH input. Please enter a number.")

def on_confirm(event):
    if selected_point is None:
        print("No point selected.")
        return
    if ph_value is None:
        print("No pH value entered.")
        return

    x, y = selected_point
    rgb = img_rgb[y, x]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[y, x]

    input_array = pd.DataFrame([{
        'R': int(rgb[0]),
        'H': int(hsv[0]),
        'V': int(hsv[2]),
        'S': int(hsv[1]),
        'G': int(rgb[1]),
        'B': int(rgb[2]),
        'pH': ph_value
    }])
    prediction = model.predict(input_array)
    print(f"Predicted Total Polyphenol Content (g/GAE): {prediction[0]}")
    ax.set_title(f"Prediction Total Polyphenol Content (g/GAE): {prediction[0]}")
    fig.canvas.draw()

# === Text box for pH input ===
axbox = plt.axes([0.35, 0.15, 0.3, 0.05])
text_box = TextBox(axbox, 'Enter pH:', initial="")
text_box.on_submit(on_submit_ph)

# === Confirm button ===
ax_confirm = plt.axes([0.4, 0.05, 0.2, 0.075])
b_confirm = Button(ax_confirm, 'Confirm')
b_confirm.on_clicked(on_confirm)

# === Click event ===
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
