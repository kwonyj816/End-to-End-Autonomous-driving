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
import glob

from cil_net_skip_connection import CIL_skip_net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 모델 인스턴스 생성
model = CIL_skip_net()
# model.load_state_dict(torch.load('/home/yongjin/venv/torch_venv/scripts/CIL/weight_straight/best_straight_4.pt'))
model.load_state_dict(torch.load('/home/yongjin/venv/torch_venv/scripts/CIL/best_straight_skip_4.pt'))
model.eval()

# 데이터 디렉토리 경로
#data_directory = '/home/yongjin/data_directory_fix_ang/test/'
data_directory = '/home/yongjin/data_directory_straight/test/'

# .json 파일 목록 가져오기
json_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.json')])

actual_acceleration = []
predicted_acceleration = []
actual_angular_velocity = []
predicted_angular_velocity = []

for json_file in json_files:
    # .json 파일 로드
    with open(os.path.join(data_directory, json_file), 'r') as f:
        data = json.load(f)

    # 이미지 base64 디코딩
    image_base64 = data['image']
    image_data = base64.b64decode(image_base64)
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # 모델 예측
    with torch.no_grad():
        # Convert image to torch tensor if needed
        image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
        linear_velocity = torch.tensor(data['status']['linear_velocity'])
        mode = torch.tensor(data['mode'])
        pred_control, img_features = model(image_tensor, linear_velocity)

    actual_acceleration.append(data['status']['acceleration'])
    predicted_acceleration.append(pred_control[0, 0].item())
    actual_angular_velocity.append(data['status']['angular_velocity'])
    predicted_angular_velocity.append(pred_control[0, 1].item())
    
    
actual_linear_velocity = np.array(actual_acceleration)
predicted_linear_velocity = np.array(predicted_acceleration)
actual_angular_velocity = np.array(actual_angular_velocity)
predicted_angular_velocity = np.array(predicted_angular_velocity)

# Plotting
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12, 6))

# Plot actual and predicted linear_velocity
plt.subplot(2, 1, 1)
plt.plot(actual_acceleration, label='Actual', marker='.')
plt.plot(predicted_acceleration, label='Predicted', marker='.')
plt.title('Acceleration Over Time')
plt.xlabel('Frame')
plt.ylabel('Acceleration')
plt.ylim(-2.0, 2.0)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='upper right',fontsize='small')

# Plot actual and predicted angular_velocity
plt.subplot(2, 1, 2)
plt.plot(actual_angular_velocity, label='Actual', marker='.')
plt.plot(predicted_angular_velocity, label='Predicted', marker='.')
plt.title('Angular Velocities Over Time')
plt.xlabel('Frame')
plt.ylabel('Angular Velocity')
plt.ylim(-1.5, 1.5)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(loc='upper right', fontsize='small')
# plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# def plot_data_in_segments(actual_data, predicted_data, segment_size, ylabel, ylim=None):
#     num_frames = len(actual_data)
#     num_plots = num_frames // segment_size + 1

#     for i in range(num_plots):
#         start_index = i * segment_size
#         end_index = min((i + 1) * segment_size, num_frames)
#         plt.figure(figsize=(12, 6))
#         plt.plot(actual_data[start_index:end_index], label='Actual', marker='o')
#         plt.plot(predicted_data[start_index:end_index], label='Predicted', marker='o')
#         plt.xlabel('Frame')
#         plt.ylabel(ylabel)
#         if ylim is not None:
#             plt.ylim(ylim)
#         plt.legend()
#         plt.show()

# # ...

# # acceleration에 대한 50프레임씩 플롯
# plot_data_in_segments(actual_acceleration, predicted_acceleration, segment_size=50, ylabel='acceleration', ylim=(-1.0, 1.0))

# # angular_velocity에 대한 50프레임씩 플롯
# plot_data_in_segments(actual_angular_velocity, predicted_angular_velocity, segment_size=50, ylabel='Angular Velocity', ylim=(-1.0, 1.0))
# %%

# 논문에서는 best_straight_4.pt 를 사용한 그래프 도시했음.