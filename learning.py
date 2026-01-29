# %%
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import json
import base64
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
import cv2
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split
from cil_net_skip_connection import CIL_skip_net 
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
class CustomDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.train = train
        self.transform = transform
        
        split_folder = 'train' if train else 'test'
        self.data_path = os.path.join(self.path, split_folder)

        self.json_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.json')])

    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_file_path = os.path.join(self.data_path, self.json_files[idx])

        with open(json_file_path, 'r') as f:
            data = json.load(f)
        image_base64 = data['image']
        image_decoded = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_decoded, dtype=np.uint8)
        image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)

        linear_velocity = data['status']['linear_velocity']
        angular_velocity = data['status']['angular_velocity']
        acceleration = data['status']['acceleration']
        mode = data['mode']
        sample = {'image': image, 'linear_velocity': linear_velocity,
                   'angular_velocity': angular_velocity,
                   'acceleration' : acceleration,
                     'mode': mode}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

path = '/home/yongjin/data_directory_straight'
full_dataset = CustomDataset(path=path, train=True, transform=transform)
test_dataset = CustomDataset(path=path, train=False, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
# %%

first_batch = next(iter(train_loader))
print("훈련 데이터 미니배치 shape:")
print("images:", first_batch['image'].shape)
print("linear_velocity:", first_batch['linear_velocity'].shape)
print("angular_velocity:", first_batch['angular_velocity'].shape)
print("acceleration:", first_batch['acceleration'].shape)
print("mode: ", first_batch['mode'].shape)

# %% 테스트
model = CIL_skip_net()
example_image = torch.randn(1,3,88,200)
example_speed = torch.randn(1,1)
control_predictions, img_features = model(example_image, example_speed)
print(control_predictions)
print(img_features.size())
print(model)
# %% shape 확인하기
for batch in train_loader:

    images = batch['image']
    linear_velocity_gt = batch['linear_velocity']
    angular_velocity_gt = batch['angular_velocity']
    acceleration = batch['acceleration']
    mode = batch['mode']

    print(acceleration)
# %% 학습 - !!!!!!!!!!!!.pt 이름바꾸기!!!!!!!!!!!!
model = CIL_skip_net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00016)
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)  # StepLR 스케줄러 설정
num_epochs = 300

best_loss = float('inf')
best_model = None
early_stopping_patience = 12
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0  # Variable to store the cumulative loss for the epoch

    for batch in train_loader:
        images = batch['image'].to(device)
        linear_velocity_gt = batch['linear_velocity'].to(device)
        angular_velocity_gt = batch['angular_velocity'].to(device)
        acceleration_gt = batch['acceleration'].to(device)
        mode = batch['mode'].to(device)

        optimizer.zero_grad()

        pred_control, _ = model(images, linear_velocity_gt)
        loss = ((pred_control[:, 0] - acceleration_gt) ** 2 + 1.1 * (pred_control[:, 1] - angular_velocity_gt) ** 2).mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # Accumulate the loss for the current batch

    scheduler.step()
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    #validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            linear_velocity_gt = batch['linear_velocity'].to(device)
            angular_velocity_gt = batch['angular_velocity'].to(device)
            acceleration_gt = batch['acceleration'].to(device)
            mode = batch['mode'].to(device)

            pred_control, _ = model(images, linear_velocity_gt)
            loss = ((pred_control[:, 0] - acceleration_gt) ** 2 + 1.1 * (pred_control[:, 1] - angular_velocity_gt) ** 2).mean()
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, val Loss: {avg_val_loss:.4f}')

    # Calculate average loss for the epoch

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model = model.state_dict()
        torch.save(best_model, 'best_straight_skip_4.pt')
        patient_counter = 0
    else:
        patient_counter += 1
        if patient_counter >= early_stopping_patience:
            print(f'Early stopping after {early_stopping_patience} epochs without improvement')
            break
# Plotting the overall loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




#%%

# Load the best model for evaluation
model = CIL_skip_net().to(device)
model.load_state_dict(torch.load('best_fix_ang_1.pt'))
model.eval()  # Set the model to evaluation mode

test_losses = []
acceleration_errors = []
angular_velocity_errors = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        linear_velocity_gt = batch['linear_velocity'].to(device)
        angular_velocity_gt = batch['angular_velocity'].to(device)
        acceleration_gt = batch['acceleration'].to(device)
        mode = batch['mode'].to(device)

        pred_control, _ = model(images, linear_velocity_gt)
        loss = ((pred_control[:, 0] - acceleration_gt) ** 2 + 1.1 * (pred_control[:, 1] - angular_velocity_gt) ** 2).mean()
        
        acceleration_errors.extend((pred_control[:, 0] - acceleration_gt).cpu().numpy())
        angular_velocity_errors.extend((pred_control[:, 1] - angular_velocity_gt).cpu().numpy())

        test_losses.append(loss.item())

# Calculate and print average test loss
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')

# Plot individual velocity errors
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(acceleration_errors)), acceleration_errors, label='Acceleration errors')
plt.title('Linear Velocity Errors')
plt.ylim(-1.0, 1.0)

plt.subplot(1, 2, 2)
plt.scatter(range(len(angular_velocity_errors)), angular_velocity_errors, label='Angular Velocity Errors')
plt.title('Angular Velocity Errors')
plt.ylim(-1.0, 1.0)

plt.show()
# %% 기록
#best_fix_ang_1.pt -> 기본
#best_fix_ang_2.pt -> 1에서 batch 128로 바꿈.
#best_fix_ang_3.pt -> 1에서 batch 64로 바꿈.
#best_fix_ang_4.pt -> 3에서 lr 0.0002로 바꿈.
#best_fix_ang_5.pt -> 3에서 lr 0.0005로 바꿈.
#best_fix_ang_6.pt -> 4에서 loss func의 가중치를 1.5로 함 -> 쓰렉
#best_fix_ang_7.pt -> 4에서 loss func의 가중치를 0.8로 함 -> not bad? 
#best_fix_ang_8.pt -> 4에서 loss func의 가중치를 1.1로 함 -> good?