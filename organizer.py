import os
import shutil
from tkinter import *
from tkinter import ttk
from PIL import Image
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    CLIPProcessor, 
    CLIPModel,
    AutoImageProcessor, 
    AutoModelForImageClassification
)
import torch
from datetime import datetime
import numpy as np
from tkinter import filedialog
from tkinter import filedialog, Toplevel, Text, END, BooleanVar, BOTH
from PIL import ImageTk

class SettingsDialog:
    def __init__(self, parent, categories):
        self.dialog = Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("400x500")
        
        self.categories = categories.copy()
        
        # Create and pack widgets
        self.frame = ttk.Frame(self.dialog, padding="10")
        self.frame.pack(fill=BOTH, expand=True)
        
        ttk.Label(self.frame, text="Custom Categories:").pack()
        
        # Category list
        self.category_text = Text(self.frame, height=20)
        self.category_text.pack(pady=10)
        self.category_text.insert('1.0', '\n'.join(self.categories))
        
        ttk.Button(self.frame, text="Save", command=self.save).pack()
        
    def save(self):
        # Update categories
        new_categories = self.category_text.get('1.0', END).strip().split('\n')
        self.categories = [cat.strip() for cat in new_categories if cat.strip()]
        self.dialog.destroy()

class ImageOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Organizer")
        self.root.geometry("500x800")  # Larger window to accommodate preview
        
        # Make the window resizable
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Create main frame with scrolling
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(N, W, E, S))
        
        # Create and pack widgets
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, pady=20)
        
        self.organize_button = ttk.Button(self.button_frame, text="Organize Images", command=self.organize_images)
        self.organize_button.grid(row=0, column=0, padx=5)
        
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", command=self.cancel_processing, state='disabled')
        self.cancel_button.grid(row=0, column=1, padx=5)
        
        self.progress = ttk.Progressbar(self.main_frame, length=300, mode='determinate')
        self.progress.grid(row=1, column=0, pady=20)
        
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=2, column=0, pady=20)
        
        # Add preview frame
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="Current Image", padding="10")
        self.preview_frame.grid(row=3, column=0, pady=20)
        
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.grid(row=0, column=0)
        
        # Add confidence display
        self.confidence_label = ttk.Label(self.preview_frame, text="")
        self.confidence_label.grid(row=1, column=0, pady=5)
        
        # Add folder selection
        self.folder_frame = ttk.Frame(self.main_frame)
        self.folder_frame.grid(row=4, column=0, pady=10)
        
        self.folder_label = ttk.Label(self.folder_frame, text="Source folder: Desktop")
        self.folder_label.grid(row=0, column=0, padx=5)
        
        self.folder_button = ttk.Button(self.folder_frame, text="Change Folder", command=self.select_folder)
        self.folder_button.grid(row=0, column=1, padx=5)
        
        self.source_folder = os.path.expanduser("~/Desktop")
        
        # Initialize models
        self.status_label.config(text="Loading AI models...")
        self.root.update()
        
        self.models = {
            'vit': {
                'processor': ViTImageProcessor.from_pretrained('google/vit-large-patch16-224'),
                'model': ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
            },
            'clip': {
                'processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                'model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            },
            'deit': {
                'processor': AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224"),
                'model': AutoModelForImageClassification.from_pretrained("facebook/deit-base-patch16-224")
            }
        }
        
        # Custom categories for CLIP model
        self.clip_categories = [
            "landscape photo", "portrait photo", "document", "screenshot",
            "meme", "artwork", "diagram", "chart", "receipt", "pet photo",
            "food photo", "selfie", "group photo", "nature photo"
        ]
        
        self.status_label.config(text="Ready to organize images")
        
        # Add settings button
        self.settings_button = ttk.Button(self.main_frame, text="Settings", command=self.show_settings)
        self.settings_button.grid(row=5, column=0, pady=10)
        
        self.stats = {
            'processed': 0,
            'errors': 0,
            'categories': {}
        }
        
        # Add theme toggle
        self.style = ttk.Style()
        self.dark_mode = BooleanVar(value=False)
        self.theme_button = ttk.Checkbutton(
            self.main_frame, 
            text="Dark Mode", 
            variable=self.dark_mode,
            command=self.toggle_theme
        )
        self.theme_button.grid(row=6, column=0, pady=5)
        
        # Add processing flag
        self.processing = False

    def classify_image(self, image):
        predictions = []
        
        # Get predictions from each model
        for model_name, model_data in self.models.items():
            try:
                if model_name == 'clip':
                    # Special handling for CLIP model
                    inputs = model_data['processor'](
                        images=image,
                        text=self.clip_categories,
                        return_tensors="pt",
                        padding=True
                    )
                    outputs = model_data['model'](**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    category_idx = probs.argmax().item()
                    category = self.clip_categories[category_idx]
                else:
                    # Standard handling for ViT and DeiT models
                    inputs = model_data['processor'](images=image, return_tensors="pt")
                    outputs = model_data['model'](**inputs)
                    predicted = outputs.logits.argmax(-1).item()
                    category = model_data['model'].config.id2label[predicted]
                
                predictions.append(category)
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
        
        # Use the most common prediction
        if predictions:
            final_category = max(set(predictions), key=predictions.count)
            confidence = len([p for p in predictions if p == final_category]) / len(predictions)
        else:
            final_category = "Uncategorized"
            confidence = 0.0
            
        return final_category, confidence

    def update_preview(self, image, category, confidence):
        try:
            # Resize image for preview
            preview_size = (200, 200)
            preview_image = image.copy()
            preview_image.thumbnail(preview_size)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(preview_image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo  # Keep a reference
            
            # Update confidence display
            self.confidence_label.config(
                text=f"Category: {category}\nConfidence: {confidence:.2%}",
                foreground='green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
            )
        except Exception as e:
            print(f"Error updating preview: {str(e)}")
            self.preview_label.configure(image='')
            self.confidence_label.config(text="Preview unavailable")

    def organize_images(self):
        self.processing = True
        self.cancel_button.configure(state='normal')
        self.organize_button.configure(state='disabled')
        
        # Reset stats
        self.stats = {'processed': 0, 'errors': 0, 'categories': {}}
        
        # Use selected folder instead of desktop
        source_folder = self.source_folder
        
        # Create dated folder in the same parent directory as source
        current_date = datetime.now().strftime("%Y-%m-%d")
        organized_folder = os.path.join(os.path.dirname(source_folder), f"Organized_Images_{current_date}")
        os.makedirs(organized_folder, exist_ok=True)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        
        # Get all image files from selected folder
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(image_extensions)]
        total_files = len(image_files)
        
        if total_files == 0:
            self.status_label.config(text="No images found on desktop")
            return
        
        self.progress['maximum'] = total_files
        
        for i, filename in enumerate(image_files):
            try:
                if not self.processing:
                    break
                    
                filepath = os.path.join(source_folder, filename)
                
                # Process image
                image = Image.open(filepath)
                category, confidence = self.classify_image(image)
                
                # Update preview
                self.update_preview(image, category, confidence)
                
                # Update stats
                self.stats['processed'] += 1
                self.stats['categories'][category] = self.stats['categories'].get(category, 0) + 1
                
                # Clean up category name
                category = category.replace("_", " ").title()
                
                # Create category folder inside dated folder
                category_folder = os.path.join(organized_folder, category)
                os.makedirs(category_folder, exist_ok=True)
                
                # Find the next available number for this category
                existing_files = os.listdir(category_folder)
                existing_numbers = []
                for f in existing_files:
                    if f.startswith(category):
                        try:
                            # Extract number from filenames like "Category_1.jpg" or "Category_1_1.jpg"
                            num = int(f.split('_')[1].split('.')[0])
                            existing_numbers.append(num)
                        except (IndexError, ValueError):
                            continue
                
                next_number = 1
                if existing_numbers:
                    next_number = max(existing_numbers) + 1
                
                # Generate new filename with sequential numbering
                base, ext = os.path.splitext(filename)
                new_filename = f"{category}_{next_number}{ext}"
                new_filepath = os.path.join(category_folder, new_filename)
                
                # Handle duplicates (just in case)
                counter = 1
                while os.path.exists(new_filepath):
                    new_filename = f"{category}_{next_number}_{counter}{ext}"
                    new_filepath = os.path.join(category_folder, new_filename)
                    counter += 1
                
                # Move and rename file
                shutil.move(filepath, new_filepath)
                
                # Update progress
                self.progress['value'] = i + 1
                self.status_label.config(text=f"Processed {i+1} of {total_files} images")
                self.root.update()
                
            except Exception as e:
                self.stats['errors'] += 1
                with open('error_log.txt', 'a') as f:
                    f.write(f"{datetime.now()}: Error processing {filename}: {str(e)}\n")
        
        # Show summary at end
        summary = f"""
        Organization complete!
        Processed: {self.stats['processed']} images
        Errors: {self.stats['errors']}
        Categories found: {', '.join(f'{k}({v})' for k,v in self.stats['categories'].items())}
        Images saved in: {organized_folder}
        """
        self.status_label.config(text=summary)
        
        self.processing = False
        self.cancel_button.configure(state='disabled')
        self.organize_button.configure(state='normal')

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder:
            self.source_folder = folder
            self.folder_label.config(text=f"Source folder: ...{folder[-30:]}")

    def show_settings(self):
        dialog = SettingsDialog(self.root, self.clip_categories)
        self.root.wait_window(dialog.dialog)
        self.clip_categories = dialog.categories

    def toggle_theme(self):
        if self.dark_mode.get():
            self.root.configure(bg='#2d2d2d')
            self.style.configure('.', background='#2d2d2d', foreground='white')
        else:
            self.root.configure(bg='#f0f0f0')
            self.style.configure('.', background='#f0f0f0', foreground='black')

    def cancel_processing(self):
        self.processing = False
        self.cancel_button.configure(state='disabled')
        self.organize_button.configure(state='normal')
        self.status_label.config(text="Processing cancelled")

if __name__ == "__main__":
    root = Tk()
    app = ImageOrganizer(root)
    root.mainloop()