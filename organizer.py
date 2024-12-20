import os
import shutil
from tkinter import *
from tkinter import ttk
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

class ImageOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Organizer")
        self.root.geometry("400x200")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(N, W, E, S))
        
        # Create and pack widgets
        self.organize_button = ttk.Button(self.main_frame, text="Organize Images", command=self.organize_images)
        self.organize_button.grid(row=0, column=0, pady=20)
        
        self.progress = ttk.Progressbar(self.main_frame, length=300, mode='determinate')
        self.progress.grid(row=1, column=0, pady=20)
        
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=2, column=0, pady=20)
        
        # Initialize the model
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def organize_images(self):
        desktop = os.path.expanduser("~/Desktop")
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        
        # Get all image files
        image_files = [f for f in os.listdir(desktop) if f.lower().endswith(image_extensions)]
        total_files = len(image_files)
        
        if total_files == 0:
            self.status_label.config(text="No images found on desktop")
            return
        
        self.progress['maximum'] = total_files
        
        for i, filename in enumerate(image_files):
            filepath = os.path.join(desktop, filename)
            
            try:
                # Process image
                image = Image.open(filepath)
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.model(**inputs)
                predicted = outputs.logits.argmax(-1).item()
                category = self.model.config.id2label[predicted]
                
                # Create category folder
                category_folder = os.path.join(desktop, category)
                os.makedirs(category_folder, exist_ok=True)
                
                # Move file
                new_filepath = os.path.join(category_folder, filename)
                if os.path.exists(new_filepath):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(new_filepath):
                        new_filename = f"{base}_{counter}{ext}"
                        new_filepath = os.path.join(category_folder, new_filename)
                        counter += 1
                
                shutil.move(filepath, new_filepath)
                
                # Update progress
                self.progress['value'] = i + 1
                self.status_label.config(text=f"Processed {i+1} of {total_files} images")
                self.root.update()
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        self.status_label.config(text="Organization complete!")

if __name__ == "__main__":
    root = Tk()
    app = ImageOrganizer(root)
    root.mainloop() 