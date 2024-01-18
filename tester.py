import tkinter as tk
import tensorflow as tf
import io
import numpy
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tkinter import filedialog
from tkinter import ttk
from PIL import EpsImagePlugin, Image, ImageFilter, ImageTk

EpsImagePlugin.gs_windows_binary = r'C:\\Program Files\\gs\\gs10.02.1\\bin\\gswin64c'


digits_model = tf.keras.models.load_model('model.keras') 
print(digits_model.input_shape)

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint App")
        self.filtering = True

        self.canvas = tk.Canvas(root, bg="red", width=600, height=600)
        self.canvas.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

        self.panel = ttk.Frame(root)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.clear_button = ttk.Button(self.panel, text="Clear Canvas", command=self.new_canvas)
        self.clear_button.pack(pady=25, padx=25)

        self.export_button = ttk.Button(self.panel, text="Import", command=self.import_image)
        self.export_button.pack(pady=25, padx=25)

        self.export_button = ttk.Button(self.panel, text="Export", command=self.export_image)
        self.export_button.pack(pady=25, padx=25)

        self.export_button = ttk.Button(self.panel, text="Random", command=self.export_image)
        # self.export_button.pack(pady=25, padx=25)

        self.c1 = ttk.Checkbutton(self.panel, text='Enable Filtering',variable=self.filtering, onvalue=True, offvalue=False)
        self.c1.pack()

        self.predict_button = ttk.Button(self.panel, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=25, padx=25)

        self.predict_text = ttk.Label(self.panel, text = "Prediction")
        self.predict_text.config(font =("Courier", 14))
        self.predict_text.pack(pady=25, padx=25)

        self.predict_num = ttk.Label(self.panel, text = "-")
        self.predict_num.config(font =("Courier", 20))
        self.predict_num.pack(pady=25, padx=25)

        self.setup_menu()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.canvas.create_rectangle(0, 0, 600, 600, outline="black", fill="black", width=25)

    def setup_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_canvas)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)


    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, outline="white", fill="white", width=50)

    def reset(self, event):
        pass

    def new_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 600, 600, outline="black", fill="black", width=25)
        self.predict_num.config(text = "-")

    def import_image(self):
        self.new_canvas()
        file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        image = Image.open(file_path)
        image = image.resize((500,500), Image.Resampling.BILINEAR)
        self.loaded_img=ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.loaded_img, anchor="nw")


    def export_image(self):
        ps = self.canvas.postscript(colormode = 'color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        size = 28,28
        if self.filtering:
            img = img.filter(ImageFilter.GaussianBlur(5))
        img.thumbnail(size, Image.Resampling.NEAREST)
        img.save("canvas_image.bmp", format="bmp")
        img.show()
        
        pixels=img.load()
        all_pixels = [[pixels[x, y][0] for x in range(28)] for y in range(28)]

        print(all_pixels)

    def predict_image(self):
        ps = self.canvas.postscript(colormode = 'color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))

        if self.filtering:
            img = img.filter(ImageFilter.GaussianBlur(10))
        size = 28,28
        img.thumbnail(size, Image.Resampling.BILINEAR)
        
        img.save("canvas_image.png", format="png")

        rimg = load_image('canvas_image.png')
        
        prediction = digits_model.predict(rimg)
        print(prediction)
        print(numpy.argmax(prediction))
        self.predict_num.config(text = numpy.argmax(prediction))


def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()