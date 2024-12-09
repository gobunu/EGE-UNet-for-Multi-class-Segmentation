import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import onnxruntime as ort



model_path = 'ege_unet.onnx'

index_to_color = {
    -1: [0, 0, 0],
    0: [0, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [0, 255, 255],
}

providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)

def process_image(image_path):

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    h, w, _ = bgr_image.shape
    if h > w:
        padding_left = (h - w) // 2
        padding_right = h - w - padding_left
        padding_top = padding_bottom = 0
    else:
        padding_top = (w - h) // 2
        padding_bottom = w - h - padding_top
        padding_left = padding_right = 0

    padded_image = cv2.copyMakeBorder(
        bgr_image, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    resized_image = cv2.resize(padded_image, (512, 512))

    image_array = np.transpose(resized_image, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    ort_inputs = {session.get_inputs()[0].name: image_array}
    ort_outs = session.run(None, ort_inputs)
    out = np.squeeze(ort_outs[0], axis=0)

    softmax_out = np.exp(out) / np.sum(np.exp(out), axis=0, keepdims=True)
    out = np.argmax(softmax_out, axis=0)

    threshold = 0.5
    softmax_out = np.max(softmax_out, axis=0)
    out[softmax_out < threshold] = 0

    height, width = out.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for index, color in index_to_color.items():
        color_image[out == index] = color

    overlay_image = copy.deepcopy(resized_image)
    overlay_image[color_image == 255] = 255

    return resized_image, overlay_image

def select_file():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")),
    )
    if file_path:

        input_image, output_image = process_image(file_path)
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(input_image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        axs[1].imshow(output_image)
        axs[1].set_title('Segmentation Output')
        axs[1].axis('off')

        canvas.figure = fig
        canvas.draw()
        plt.close(fig)

root = tk.Tk()
root.title("Biomedical Image Processing")

fig = plt.figure(figsize=(20, 10))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, padx=5, pady=5)

button_font = ('Arial', 20)

ttk.Button(root, text="Select File", command=select_file, style="TButton").grid(row=1, column=1, padx=10, pady=10)

style = ttk.Style()
style.configure("TButton", font=button_font)

root.mainloop()
