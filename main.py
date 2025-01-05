import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.linalg import svd

def compute_svd(image_array):
    """
    Compute the reduced SVD of the image's pixel matrix.

    Parameters:
        image_array (numpy.ndarray): The pixel data of the image.

    Returns:
        U, S, Vt: Matrices from the SVD decomposition.
    """
    # Compute the SVD of the image matrix
    U, S, Vt = svd(image_array, full_matrices=False)
    print("SVD Computation Results:")
    print(f"U matrix shape: {U.shape}")
    print(f"Singular values shape: {S.shape}")
    print(f"Vᵀ matrix shape: {Vt.shape}")
    return U, S, Vt

def load_and_display_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    if file_path:
        try:
            # Open the image and resize it to fit within the canvas
            image = Image.open(file_path)
            image = image.resize((400, 400), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image)

            # Display the image on the canvas
            canvas.image = img_tk  # Keep a reference to prevent garbage collection
            canvas.create_image(200, 200, image=img_tk)

            # Convert the image to grayscale for SVD
            gray_image = image.convert("L")  # Convert to grayscale
            image_array = np.array(gray_image)

            # Compute the SVD
            compute_svd(image_array)
        except Exception as e:
            print(f"Error loading image: {e}")

# Create the main application window
root = tk.Tk()
root.title("LAA TSVD")

# Create a canvas to display the image
canvas = tk.Canvas(root, width=400, height=400, bg="gray")
canvas.pack(pady=20)

# Create a button to load an image
btn_load_image = tk.Button(root, text="Load Image", command=load_and_display_image)
btn_load_image.pack(pady=10)

# Run the application
root.mainloop()