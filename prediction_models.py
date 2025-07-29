import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.training_history = {'loss': [], 'accuracy': []}
        
    def train_model(self, X_train, y_train, epochs=100, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = self(X_train)
                accuracy = accuracy_score(y_train.numpy().argmax(axis=1), 
                                        predictions.numpy().argmax(axis=1))
            
            self.training_history['loss'].append(loss.item())
            self.training_history['accuracy'].append(accuracy)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X)
            return self(X).numpy()
    
    def plot_metrics(self, save_path):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(save_path + '_metrics.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path + '_confusion.png')
        plt.close()

class RNNModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__(input_size, hidden_size, output_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class CNNModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNModel, self).__init__(input_size, hidden_size, output_size)
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3)
        
        # Calculate the size of flattened features
        self.flatten_size = self._get_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, input_size, input_size)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        return x.numel()
    
    def forward(self, x):
        x = x.transpose(1, 2) if len(x.shape) == 3 else x.unsqueeze(0).transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x