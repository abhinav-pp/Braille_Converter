import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np

# Model definition (same as previous example)
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

# Training and validation loop with error handling for data loading
num_classes = 26
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and data augmentation (assumes consistent transformations)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

avg_f1=0
count=0

# Load the dataset and create DataLoaders
dataset_path = r'C:\Users\abhin\Desktop\mini_project\Mini_Project\Braille Dataset'
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model and load the pre-trained state dict
model = CharacterRecognitionModel(num_classes)
model.to(device)

try:
    state_dict = torch.load(r'C:\Users\abhin\Desktop\mini_project\Mini_Project\model\braille_recognition_model_final.pth')
    model.load_state_dict(state_dict, strict=False)
except RuntimeError as e:
    print("Error loading state dict:", e)

model.eval()  # Set to evaluation mode for validation

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()  # Classification loss function
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # AdamW optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)  # Learning rate scheduler

# Training loop with validation and F1 score calculation
num_epochs = 50  # Adjust based on the model's performance and complexity
best_loss = float('inf')
patience = 10
no_improvement_count = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set to training mode
    epoch_loss = 0.0
    
    for batch in train_dataloader:
        if len(batch) == 2:  # Ensure only two elements (inputs, labels)
            inputs, labels = batch
        else:
            inputs = batch[0]
            labels = batch[1]
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Model update
        
        epoch_loss += loss.item() * inputs.size(0)  # Accumulate epoch loss
    
    epoch_loss = epoch_loss / len(train_dataloader.dataset)  # Average epoch loss
    
    # Validation phase
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():  # No gradient computation during validation
        for batch in val_dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = batch[1]
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            predicted = torch.argmax(outputs, 1)  # Get the predicted class
            
            y_true.extend(labels.cpu().numpy())  # Collect true labels
            y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels
            
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
    
    val_loss = val_loss / len(val_dataset)  # Average validation loss
    
    # Calculate accuracy and F1 score
    val_accuracy = 100 * (sum(1 for a, b in zip(y_pred, y_true) if a == b) / len(y_true))  # Calculate accuracy
    f1 = f1_score(y_true, y_pred, average='macro')  # Macro-average for multi-class F1
    avg_f1=avg_f1+f1
    
    # Output validation loss, accuracy, and F1 score
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, F1 Score: {f1:.4f}")
    count=count+1
    
    # Learning rate adjustment based on validation loss
    scheduler.step(val_loss)  
    
    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print("Early stopping...")
            break
    
print((avg_f1)/count)
# Save the trained model's state dict
torch.save(model.state_dict(), r'C:\Users\abhin\Desktop\mini_project\Mini_Project\model\braille_recognition_model_final.pth')  # Save the model