import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_svd_pruning():
    # 1. 입력 데이터 생성: 단위 원 (Unit Circle)
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    input_points = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32) # (100, 2)

    # 2. 임의의 가중치 행렬 W 설정 (2x2 FC Layer)
    # 기하학적 효과를 극적으로 보여주기 위해 상관관계가 있는 행렬 생성
    W = torch.tensor([[2.0, 1.0],
                      [1.5, 3.0]], dtype=torch.float32)

    # 3. 원본 아핀 변환 (y = Wx) *편의상 bias는 생략
    output_original = input_points @ W.T

    # 4. SVD 수행 및 시각화
    U, S, Vh = torch.linalg.svd(W)
    
    print(f"Original Singular Values: {S}")
    # 예: tensor([3.85..., 1.15...]) -> 큰 축과 작은 축의 비율 확인

    # 5. Low-Rank Approximation (Rank-1 근사)
    # 가장 작은 특이값을 0으로 만듦 (정보 손실 유도)
    S_pruned = S.clone()
    S_pruned[-1] = 0  # 가장 작은 특이값 제거 (Min Singular Value -> 0)
    
    # 근사된 행렬 W_approx 재구성
    # W_approx = U @ diag(S_pruned) @ Vh
    W_approx = U @ torch.diag(S_pruned) @ Vh

    # 6. 근사된 아핀 변환 적용
    output_pruned = input_points @ W_approx.T

    # --- 시각화 ---
    plt.figure(figsize=(12, 5))

    # Plot 1: 입력 공간 (단위 원)
    plt.subplot(1, 3, 1)
    plt.scatter(input_points[:, 0], input_points[:, 1], c=theta, cmap='hsv', s=10)
    plt.title("1. Input Space (Unit Circle)")
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Plot 2: 원본 FC Layer 출력
    plt.subplot(1, 3, 2)
    plt.scatter(output_original[:, 0], output_original[:, 1], c=theta, cmap='hsv', s=10)
    plt.title("2. Original Output (Ellipse)\nFull Rank")
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # 주축(Principal Axis) 표시 (U 벡터 방향)
    # 가장 큰 특이값 방향
    axis_major = U[:, 0] * S[0]
    plt.arrow(0, 0, axis_major[0], axis_major[1], head_width=0.2, color='black', label='Major Axis')

    # Plot 3: 가지치기된 FC Layer 출력
    plt.subplot(1, 3, 3)
    plt.scatter(output_pruned[:, 0], output_pruned[:, 1], c=theta, cmap='hsv', s=10)
    plt.title("3. Pruned Output (Line)\nRank-1 Approximation")
    plt.grid(True)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # 제거된 축 설명
    plt.text(0, -2, "Minor Axis Collapsed", ha='center', color='red')

    plt.tight_layout()
    plt.savefig('svd_pruning_visualization.png')
    print("Visualization saved to svd_pruning_visualization.png")
    # plt.show()

# 실험 실행
visualize_svd_pruning()