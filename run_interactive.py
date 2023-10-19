import tkinter as tk
from predict import *
import pyautogui
import os
from train_network import forward_propagation
import numpy as np
from PIL import Image

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.old_x = None
        self.old_y = None

        # Canvas to draw on
        self.canvas = tk.Canvas(root, bg="black", width=1120, height=1120)
        self.canvas.grid(row=0, column=0, columnspan=3, pady=20)  # Use grid for layout

        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        clear_button = tk.Button(root, text="Clear", command=self.clear_canvas, width=10, height=2)
        clear_button.grid(row=1, column=0)

        confirm_button = tk.Button(root, text="Confirm", command=self.confirm_image, width=10, height=2)
        confirm_button.grid(row=1, column=1)

        exit_button = tk.Button(root, text="Exit", command=self.root.destroy, width=10, height=2)
        exit_button.grid(row=1, column=2)

    def start_draw(self, event):
        self.old_x, self.old_y = event.x, event.y

    def draw(self, event):
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line((self.old_x, self.old_y, x, y), width=70, capstyle=tk.ROUND, smooth=tk.TRUE, fill='white')
            self.old_x, self.old_y = x, y


    def stop_draw(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")

    def confirm_image(self):
        img = self.get_image_from_canvas()
        print('detected ',predict(r'temp.jpg'))
        return
    def get_image_from_canvas(self):
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        
        screenshot = pyautogui.screenshot(region=(x0, y0, x1 - x0, y1 - y0))
        screenshot_resized = screenshot.resize((28, 28))
        gray_scaled_img = screenshot_resized.convert('L')
        gray_scaled_img.save('temp.jpg')
        return gray_scaled_img

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Handwritten Digit Recognition")
    app = DrawingApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    os.remove("temp.jpg")
