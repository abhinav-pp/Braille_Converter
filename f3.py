import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
from PIL import Image
import cv2
import numpy as np

# Define the CharacterRecognitionModel
class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CharacterRecognitionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # New convolutional layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # New convolutional layer
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # Additional new convolutional layer
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 512),  # Adjust based on the feature extraction output size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # Additional linear layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Device setup and load pre-trained state dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterRecognitionModel(26)
model.to(device)

# Load the pre-trained state dictionary
try:
    state_dict = torch.load(r'C:\Users\abhin\Desktop\mini_project\Mini_Project\model\braille_recognition_model_final.pth')
    model.load_state_dict(state_dict, strict=False)
except RuntimeError as e:
    print("Error loading state dict:", e)

model.eval()  # Set to evaluation mode

# Braille-to-English dictionary
braille_to_english = {i: chr(97 + i) for i in range(26)}  # 'a' to 'z'

# Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Function to ensure grayscale
def ensure_grayscale(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Function to preprocess the image
def preprocess_image(image):
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Convert to grayscale if necessary
        grayscale_image = ensure_grayscale(image_np)

        return Image.fromarray(grayscale_image)

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Function to resize the image to the expected size
def resize_image(image, new_size):
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)  # High-quality resampling
    return resized_image

# Function to segment Braille characters
def segment_braille_characters(image, num_columns):
    width, height = image.size
    segment_width = width // num_columns
    
    braille_images = []
    for i in range(num_columns):
        left = i * segment_width
        right = (i + 1) * segment_width
        cropped_image = image.crop((left, 0, right, height))
        braille_images.append(cropped_image)
    
    return braille_images

# Function to convert Braille images to text
def braille_to_text(braille_images, model, transform, braille_to_english):
    text = ""
    for img in braille_images:
        transformed_img = transform(img)
        input_tensor = transformed_img.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
        
        predicted_class = torch.argmax(output, dim=1).item()
        
        text += braille_to_english[predicted_class]
    
    return text

# Function to convert text into sentences
def create_sentences(text):
    # In this context, this function could create more structured sentences if needed.
    return text

# Example usage
image_path = sys.argv[1]
num_columns = int(sys.argv[2])  # Number of Braille columns in the input image

original_image = Image.open(image_path)

# Resize image to fit expected input size
resized_image = resize_image(original_image, (num_columns * 28, 28))

# Preprocess and segment the image
preprocessed_image = preprocess_image(resized_image)  # Convert to grayscale

if preprocessed_image is not None:
    braille_images = segment_braille_characters(preprocessed_image, num_columns)
    text = braille_to_text(braille_images, model, transform, braille_to_english)
    sentence = create_sentences(text)
    print("Converted sentence:", sentence)
else:
    print("Error: Preprocessing failed. Check the input image.")
