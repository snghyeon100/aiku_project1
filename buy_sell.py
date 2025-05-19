import torch
from models.mlplob import MLPLOB
import yaml
import pandas as pd
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. config.yaml 불러와서 하이퍼파라미터 세팅
config_path = "outputs/2025-05-16/05-10-59/.hydra/config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 2. config 내부 구조 분해
cfg_model = config['model']['hyperparameters_fixed']
cfg_data = config['dataset']

dim = cfg_model['hidden_dim']
depth = cfg_model['num_layers']
seq_size = cfg_model['seq_size']
num_features = 40  # 보통 고정값이거나 따로 config에 정의되어 있지 않음
dataset_type = cfg_data['type']

# 3. 모델 정의
model = MLPLOB(
    hidden_dim=dim,
    num_layers=depth,
    seq_size=seq_size,
    num_features=num_features,
    dataset_type=dataset_type
)

# 4. 모델 가중치 불러오기
weight_path = "saved_models/MLPLOB/BTC_seq_size_384_horizon_10_seed_1/model.pt"
model.load_state_dict(torch.load(weight_path, map_location="cpu"))
model.to(device)
model.eval()

# 5. CSV 오더북 데이터 불러오기
csv_path = r"data\BINANCE\BINANCE_2025-05-01_2025-05-10\BINANCE_2025-05-10_34200000_57600000_orderbook.csv"
df = pd.read_csv(csv_path)

# 6. 입력 시퀀스 추출
assert df.shape[1] == num_features, f"CSV의 feature 수 ({df.shape[1]})가 모델 입력과 다릅니다: {num_features}"
assert len(df) >= seq_size, f"데이터 수가 부족합니다. 최소 {seq_size}개의 row 필요."

snapshot_seq = df.tail(seq_size).values  # (seq_size, num_features)
input_tensor = torch.tensor(snapshot_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_size, num_features)
input_tensor = input_tensor.to(device)
# 7. 예측 실행
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
    pred_class = torch.argmax(probs).item()

# 8. 결과 출력
classes = ['상승 (Up)', '유지 (Stable)', '하락 (Down)']
print(f"📊 예측 확률: {probs.tolist()}")
print(f"✅ 예측 결과: {classes[pred_class]} (클래스 {pred_class})")
