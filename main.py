import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.linalg import svd

# Global variables to store images and their current zoom factor
original_images = []  # Store original images
current_zoom = 0.2    # Initial zoom factor

def compute_svd(image_array):
    U, S, Vt = svd(image_array, full_matrices=False)
    print("\n--- Full SVD Matrices ---")
    print(f"U matrix (shape: {U.shape}):\n{U}")
    print(f"Singular values (S, shape: {S.shape}):\n{S}")
    print(f"V^T matrix (shape: {Vt.shape}):\n{Vt}")
    return U, S, Vt


def reconstruct_image(U, S, Vt, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))
    reconstructed = np.clip(reconstructed, 0, 255)  # Clip values to valid range
    print(f"\n--- Reconstructed Matrix for k = {k} ---")
    print(f"\n File size estimate: {calculate_file_size(U, S, Vt, k)}")
    return reconstructed

def calculate_file_size(U, S, Vt, k):
    """
    Estimate the file size based on the size of the matrices used.
    """
    rows, cols = U.shape[0], Vt.shape[1]
    size = rows * k + k + cols * k  # U_k, S_k, and Vt_k sizes
    return size  # Assuming 1 byte per float


def display_images(original_image, full_svd_image, reconstructions, k_values):
    """
    Display images in two rows:
    Row 1: Original image repeated 4 times.
    Row 2: Full SVD, k1, k2, k3 reconstructions.
    """
    global original_images, current_zoom
    canvas.delete("all")  # Clear all items from the canvas
    canvas.image_refs = []  # Store all image references to prevent garbage collection

    # Store original images for zooming functionality
    original_images = [original_image] * 4 + \
                      [Image.fromarray(full_svd_image.astype(np.uint8))] + \
                      [Image.fromarray(img.astype(np.uint8)) for img in reconstructions]
    labels = ["Original"] * 4 + ["Full SVD"] + [f"k = {k}" for k in k_values]

    render_images(labels)


def render_images(labels):
    """
    Render images on the canvas in two rows based on the current zoom level.
    """
    global original_images, current_zoom
    canvas.delete("all")  # Clear all items before re-rendering
    resize_factor = current_zoom  # Apply current zoom factor
    canvas.image_refs = []  # Reset image references

    row_padding = int(300 * current_zoom)  # Base padding scaled by zoom
    x_offset = 10  # Horizontal offset

    # Calculate the maximum height of the resized images in Row 1
    max_image_height_row1 = max(int(img.height * resize_factor) for img in original_images[:4])

    # Determine row positions dynamically
    row1_y = 100  # Starting position for Row 1
    row2_y = row1_y + max_image_height_row1 + row_padding  # Position for Row 2 below Row 1

    # Iterate over images and labels
    for i, (img, label) in enumerate(zip(original_images, labels)):
        resized_img = img.resize(
            (int(img.width * resize_factor), int(img.height * resize_factor)),
            Image.Resampling.LANCZOS
        )
        img_tk = ImageTk.PhotoImage(resized_img)

        # Determine row and position
        row_y = row1_y if i < 4 else row2_y  # Use Row 1 or Row 2 vertical position
        x = x_offset + resized_img.width // 2

        canvas.create_image(x, row_y, image=img_tk, anchor="center")
        canvas.image_refs.append(img_tk)  # Append to list
        canvas.create_text(x, row_y + resized_img.height // 2 + 20, text=label, fill="white")

        x_offset += resized_img.width + 40  # Move to the next image position
        if (i + 1) % 4 == 0:  # Reset horizontal offset for new row
            x_offset = 10

    canvas.config(scrollregion=canvas.bbox("all"))  # Update scrollable area


def load_and_process_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        try:
            image = Image.open(file_path)
            gray_image = image.convert("L")  # Convert to grayscale
            image_array = np.array(gray_image)

            print("\n--- Original Pixel Matrix ---")
            print(image_array)

            U, S, Vt = compute_svd(image_array)

            # Full SVD reconstruction
            reconstructed_full = reconstruct_image(U, S, Vt, len(S))  # Use all singular values

            # Get user-defined k values
            k1 = int(entry_k1.get())
            k2 = int(entry_k2.get())
            k3 = int(entry_k3.get())
            k_values = [k1, k2, k3]

            # TSVD reconstructions
            reconstructions = [reconstruct_image(U, S, Vt, k) for k in k_values]

            # Display images
            display_images(image, reconstructed_full, reconstructions, k_values)

        except Exception as e:
            print(f"Error loading or processing image: {e}")


def zoom_in():
    """
    Zoom in by increasing the zoom factor and re-rendering the images.
    """
    global current_zoom
    current_zoom *= 1.1  # Increase zoom factor
    labels = ["Original"] * 4 + ["Full SVD"] + [f"k = {k}" for k in [int(entry_k1.get()), int(entry_k2.get()), int(entry_k3.get())]]
    render_images(labels)


def zoom_out():
    """
    Zoom out by decreasing the zoom factor and re-rendering the images.
    """
    global current_zoom
    current_zoom *= 0.9  # Decrease zoom factor
    labels = ["Original"] * 4 + ["Full SVD"] + [f"k = {k}" for k in [int(entry_k1.get()), int(entry_k2.get()), int(entry_k3.get())]]
    render_images(labels)


def zoom_in_5x():
    """
    Zoom in 5x by increasing the zoom factor and re-rendering the images.
    """
    global current_zoom
    current_zoom *= (1.1 ** 5)  # Increase zoom factor by 5x
    labels = ["Original"] * 4 + ["Full SVD"] + [f"k = {k}" for k in [int(entry_k1.get()), int(entry_k2.get()), int(entry_k3.get())]]
    render_images(labels)


def zoom_out_5x():
    """
    Zoom out 5x by decreasing the zoom factor and re-rendering the images.
    """
    global current_zoom
    current_zoom *= (0.9 ** 5)  # Decrease zoom factor by 5x
    labels = ["Original"] * 4 + ["Full SVD"] + [f"k = {k}" for k in [int(entry_k1.get()), int(entry_k2.get()), int(entry_k3.get())]]
    render_images(labels)


def start_pan(event):
    """
    Start panning the canvas.
    """
    canvas.scan_mark(event.x, event.y)


def pan(event):
    """
    Pan the canvas using the mouse.
    """
    canvas.scan_dragto(event.x, event.y, gain=1)


# Create the main application window
root = tk.Tk()
root.title("LAA TSVD")

# Set the window to fullscreen
root.attributes("-fullscreen", True)

# Add a way to exit fullscreen
def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)

# Bind the Escape key to exit fullscreen mode
root.bind("<Escape>", exit_fullscreen)

# Create a scrollable canvas
canvas = tk.Canvas(root, bg="black")
canvas.pack(fill="both", expand=True)

# Add scrollbars
hbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
hbar.pack(side="bottom", fill="x")
vbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
vbar.pack(side="right", fill="y")
canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

# Bind pan events
canvas.bind("<ButtonPress-1>", start_pan)  # Start panning on left mouse button press
canvas.bind("<B1-Motion>", pan)  # Pan while dragging

# Add input fields for k values
frame = tk.Frame(root)
frame.pack(pady=10)
tk.Label(frame, text="k1:").grid(row=0, column=0)
entry_k1 = tk.Entry(frame, width=5)
entry_k1.grid(row=0, column=1)
entry_k1.insert(0, "10")

tk.Label(frame, text="k2:").grid(row=0, column=2)
entry_k2 = tk.Entry(frame, width=5)
entry_k2.grid(row=0, column=3)
entry_k2.insert(0, "50")

tk.Label(frame, text="k3:").grid(row=0, column=4)
entry_k3 = tk.Entry(frame, width=5)
entry_k3.grid(row=0, column=5)
entry_k3.insert(0, "100")

# Add a button to load an image
btn_load_image = tk.Button(root, text="Load Image", command=load_and_process_image)
btn_load_image.pack(pady=10)

# Add Zoom In, Zoom Out, and 5x Zoom Buttons
zoom_frame = tk.Frame(root)
zoom_frame.pack(pady=10)
btn_zoom_in_5x = tk.Button(zoom_frame, text="5x +", command=zoom_in_5x)
btn_zoom_in_5x.grid(row=0, column=0, padx=10)
btn_zoom_in = tk.Button(zoom_frame, text="+", command=zoom_in)
btn_zoom_in.grid(row=0, column=1, padx=10)
btn_zoom_out = tk.Button(zoom_frame, text="-", command=zoom_out)
btn_zoom_out.grid(row=0, column=2, padx=10)
btn_zoom_out_5x = tk.Button(zoom_frame, text="5x -", command=zoom_out_5x)
btn_zoom_out_5x.grid(row=0, column=3, padx=10)

root.mainloop()
