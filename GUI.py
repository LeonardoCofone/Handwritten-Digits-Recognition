import os
import sys
import io
import numpy as np
from PIL import Image as PILImage, UnidentifiedImageError, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle


if sys.platform not in ('android', 'ios'):
    Window.size = (600, 500)
    Window.clearcolor = (0.9, 0.9, 0.95, 1)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path(os.path.join('models', 'mnist_transfer_model.h5'))

class LoadingScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)

        self.label_top = Label(
            text="Loading AI model...\nPlease wait a moment",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_top.bind(size=self.label_top.setter('text_size'))
        self.add_widget(self.label_top)

        self.loading_image = KivyImage(
            source=resource_path('assets/LOGO.png'),
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.6)
        )
        self.add_widget(self.loading_image)

        self.label_bottom = Label(
            text="Powered by Leonardo Cofone",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_bottom.bind(size=self.label_bottom.setter('text_size'))
        self.add_widget(self.label_bottom)


class MNISTApp(App):
    def build(self):
        self.title = "Handwritten Digit Classifier - Leonardo Cofone"
        self.root_layout = BoxLayout(orientation='vertical')
        self.loading_screen = LoadingScreen()
        self.root_layout.add_widget(self.loading_screen)

        Clock.schedule_once(lambda dt: self.load_model(), 1)
        return self.root_layout

    def load_model(self):
        try:
            self.model = load_model(MODEL_PATH)
            self.root_layout.clear_widgets()
            self.show_main_screen()
        except Exception as e:
            self.loading_screen.label_top.text = f"Error loading model:\n{e}"
            self.model = None

    def show_main_screen(self):
        Window.clearcolor = (0.75, 0.87, 1, 1)

        self.title_label = Label(
            text="Handwritten Digit Classifier",
            font_size='24sp',
            size_hint=(1, 0.1),
            color=(1, 1, 1, 1),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        with self.title_label.canvas.before:
            Color(0.02, 0.53, 0.82, 1)
            self.rect = Rectangle(pos=self.title_label.pos, size=self.title_label.size)
        self.title_label.bind(pos=self.update_rect, size=self.update_rect)
        self.root_layout.add_widget(self.title_label)

        self.image_panel = KivyImage(
            size_hint=(1, 0.6),
            allow_stretch=True,
            keep_ratio=True
        )
        self.root_layout.add_widget(self.image_panel)

        self.result_label = Label(
            text="Upload a handwritten digit image",
            font_size='18sp',
            size_hint=(1, 0.15),
            color=(1, 1, 1, 1),
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        with self.result_label.canvas.before:
            Color(0.02, 0.53, 0.82, 1)
            self.bottom_rect = Rectangle(pos=self.result_label.pos, size=self.result_label.size)
        self.result_label.bind(pos=self.update_bottom_rect, size=self.update_bottom_rect)
        self.root_layout.add_widget(self.result_label)

        button_layout = BoxLayout(size_hint=(1, 0.15), spacing=10, padding=10)
        with button_layout.canvas.before:
            Color(0.02, 0.53, 0.82, 1)
            self.button_rect = Rectangle(pos=button_layout.pos, size=button_layout.size)
        button_layout.bind(pos=self.update_button_rect, size=self.update_button_rect)

        self.load_button = Button(
            text="Upload Image",
            color=(1, 1, 1, 1),
            background_color=(0.1, 0.5, 0.8, 1)
        )
        self.load_button.bind(on_release=self.open_file_chooser)

        self.reset_button = Button(
            text="Reset",
            color=(1, 1, 1, 1),
            background_color=(0.2, 0.6, 0.9, 1)
        )
        self.reset_button.bind(on_release=self.reset_app)

        button_layout.add_widget(self.load_button)
        button_layout.add_widget(self.reset_button)
        self.root_layout.add_widget(button_layout)


    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_bottom_rect(self, instance, value):
        self.bottom_rect.pos = instance.pos
        self.bottom_rect.size = instance.size

    def update_button_rect(self, instance, value):
        self.button_rect.pos = instance.pos
        self.button_rect.size = instance.size


    def open_file_chooser(self, instance):
        home = os.path.expanduser("~")
        downloads = os.path.join(home, 'Downloads')
        pictures = os.path.join(home, 'Pictures')

        if os.path.exists(pictures):
                initial_path = pictures
        elif os.path.exists(downloads):
                initial_path = downloads
        else:
                initial_path = home

        content = FileChooserIconView(
            path=initial_path, filters=['*.jpg', '*.jpeg', '*.png', '*.bmp'])
        popup = Popup(
            title="Select a digit image",
            content=content,
            size_hint=(0.9, 0.9),
            auto_dismiss=False
        )

        def on_submit(instance, selection, touch):
            if selection:
                popup.dismiss()
                self.load_and_predict(selection[0])

        content.bind(on_submit=on_submit)
        popup.open()

    def load_and_predict(self, filepath):
        try:
            pil_img = PILImage.open(filepath).convert('RGB')
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.resize((128, 128), PILImage.Resampling.LANCZOS)

            img_array = np.array(pil_img)              
            img_array = img_array.astype('float32') / 255.0  
            img_array = np.expand_dims(img_array, axis=0) 

            preds = self.model.predict(img_array)
            pred_class = np.argmax(preds)
            confidence = preds[0][pred_class]

            disp_img = pil_img.resize((280, 280), PILImage.Resampling.LANCZOS)
            data = io.BytesIO()
            disp_img.save(data, format='png')
            data.seek(0)
            self.image_panel.texture = self.load_texture(data)

            self.result_label.text = f"Predicted digit: {pred_class}\nConfidence: {confidence:.2%}"
            self.reset_button.disabled = False

        except UnidentifiedImageError:
            self.result_label.text = "Image format not supported."
            self.reset_button.disabled = True
        except Exception as e:
            self.result_label.text = f"Error: {e}"
            self.reset_button.disabled = True

    def load_texture(self, data):
        from kivy.core.image import Image as CoreImage
        return CoreImage(data, ext='png').texture

    def reset_app(self, instance):
        self.image_panel.texture = None
        self.result_label.text = "Upload a handwritten digit image"

if __name__ == "__main__":
    MNISTApp().run()

