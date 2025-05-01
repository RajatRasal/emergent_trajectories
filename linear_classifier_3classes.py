import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

import time
from torch.utils.data import random_split

from dataset import classifier_ds


class MLP(nn.Module):

    def __init__(self, input_dim, output_dims):
        super().__init__()

        self.output_fc0 = nn.Linear(input_dim, output_dims[0])
        self.output_fc1 = nn.Linear(input_dim, output_dims[1])
        self.output_fc2 = nn.Linear(input_dim, output_dims[2])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:,:3,:,:].reshape(batch_size, -1)

        y_pred = {}
        y_pred[0] = self.output_fc0(x)
        y_pred[1] = self.output_fc1(x)
        y_pred[2] = self.output_fc2(x)

        return y_pred 


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = {0: 0, 1: 0, 2: 0}
    model.train()

    for (x, y) in tqdm(iterator, desc="Training"):
        x = x.to(device)
        y = [_y.to(device) for _y in y]
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred[0], y[0]) + criterion(y_pred[1], y[1]) + criterion(y_pred[2], y[2])
        acc = {}
        acc[0] = calculate_accuracy(y_pred[0], y[0])
        acc[1] = calculate_accuracy(y_pred[1], y[1])
        acc[2] = calculate_accuracy(y_pred[2], y[2])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc[0] += acc[0].item()
        epoch_acc[1] += acc[1].item()
        epoch_acc[2] += acc[2].item()

    epoch_acc[0] /= len(iterator)
    epoch_acc[1] /= len(iterator)
    epoch_acc[2] /= len(iterator)

    return epoch_loss / len(iterator), epoch_acc 


def calc_mse(pred, gt): 
    return torch.sqrt(torch.mean((pred-gt)**2))
    

@torch.no_grad()
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = {0: 0, 1: 0, 2: 0}

    model.eval()
    for (x, y) in tqdm(iterator, desc="Evaluating"):
        x = x.to(device)
        y = [_y.to(device) for _y in y]
        y_pred = model(x)
        loss = criterion(y_pred[0], y[0]) + criterion(y_pred[1], y[1]) + criterion(y_pred[2], y[2])
        acc = {}
        acc[0] = calculate_accuracy(y_pred[0], y[0])
        acc[1] = calculate_accuracy(y_pred[1], y[1])
        acc[2] = calculate_accuracy(y_pred[2], y[2])
        epoch_loss += loss.item()
        epoch_acc[0] += acc[0].item()
        epoch_acc[1] += acc[1].item()
        epoch_acc[2] += acc[2].item()

    epoch_acc[0] /= len(iterator)
    epoch_acc[1] /= len(iterator)
    epoch_acc[2] /= len(iterator)

    return epoch_loss / len(iterator), epoch_acc


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__": 
    classifier_path = "./models/classifier/shapes/"
    os.makedirs(classifier_path, exist_ok=True)
    best_model_path = os.path.join(classifier_path, "model.pt")

    data_path = "./datasets/single-body_2d_3classes 2"
    pixel_size = 28
    train_ds, test_ds = classifier_ds(data_path, pixel_size)
    train_ds, val_ds = random_split(train_ds, [0.9, 0.1])

    BATCH_SIZE = 128
    train_iterator = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
    val_iterator = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE)
    test_iterator = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    INPUT_DIM = pixel_size * pixel_size * 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIMS = [2, 2, 2]
    model = MLP(INPUT_DIM, OUTPUT_DIMS)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    EPOCHS = 10
    best_valid_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
    
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion, device)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
    
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc[0] * 100:.2f}% {train_acc[1] * 100:.2f}% {train_acc[2] * 100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc[0] * 100:.2f}% {valid_acc[1] * 100:.2f}% {valid_acc[2] * 100:.2f}%')

    model.load_state_dict(torch.load(best_model_path, weights_only=True)) 
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc[0]*100:.2f}% {test_acc[1]*100:.2f}% {test_acc[2]*100:.2f}%')
