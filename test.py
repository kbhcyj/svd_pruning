"""
SVD 기반 네트워크 가지치기 시각화

본 코드는 딥러닝 모델 경량화 기술인 네트워크 가지치기(Network Pruning)의
원리를 SVD(특이값 분해)를 통해 기하학적으로 시각화합니다.

주요 개념:
- 완전 연결 계층(FC Layer)의 가중치 행렬을 SVD로 분해
- 작은 특이값을 제거하여 Low-Rank Approximation 수행
- 이는 모델의 표현 능력(Rank)을 감소시켜 파라미터를 줄이는 것과 동일
- 2차원 공간에서의 변환을 시각화하여 정보 손실을 직관적으로 이해
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_svd_pruning():
    # ============================================================
    # 1. 입력 데이터 생성: 단위 원 (Unit Circle)
    # ============================================================
    # 신경망의 입력 분포를 대변하기 위해 모든 방향성을 가진 단위 벡터들을 생성
    # 2차원 공간의 모든 방향을 균등하게 샘플링하여 가중치 행렬이
    # 입력을 어떻게 변형시키는지 관찰할 수 있음
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    input_points = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32) # (100, 2)

    # ============================================================
    # 2. 임의의 가중치 행렬 W 설정 (2x2 FC Layer)
    # ============================================================
    # 이 행렬은 입력 차원 2 -> 출력 차원 2의 완전 연결 계층을 나타냄
    # 실제 신경망에서는 수천~수만 차원일 수 있지만, 원리는 동일함
    # 
    # 네트워크 가지치기 관점:
    # - 이 행렬의 랭크(Rank)는 2 (풀 랭크)
    # - SVD로 분해하면 2개의 특이값(σ₁, σ₂)을 얻음
    # - 작은 특이값(σ₂)을 0으로 만들면 Rank-1 행렬이 됨
    # - 이는 모델의 표현력을 의도적으로 감소시켜 파라미터를 압축하는 것
    W = torch.tensor([[2.0, 1.0],
                      [1.5, 3.0]], dtype=torch.float32)

    # ============================================================
    # 3. 원본 아핀 변환 (y = Wx) *편의상 bias는 생략
    # ============================================================
    # FC Layer의 순전파(Forward Pass): y = Wx + b
    # 가지치기 전의 원본 모델이 입력을 어떻게 변환하는지 계산
    # 단위 원이 타원(Ellipse)으로 변형됨 -> 2차원 정보 모두 보존
    output_original = input_points @ W.T

    # ============================================================
    # 4. SVD 수행: W = U @ Σ @ V^T
    # ============================================================
    # 특이값 분해(SVD)는 행렬을 다음과 같이 분해:
    # W = U @ diag(S) @ V^T
    # 
    # 해석:
    # - U: 출력 공간의 회전 (Left Singular Vectors)
    # - S: 각 주성분의 중요도 (Singular Values, 특이값)
    # - V^T: 입력 공간의 회전 (Right Singular Vectors)
    # 
    # 네트워크 가지치기 관점:
    # - 특이값 S는 각 "정보 채널"의 중요도를 나타냄
    # - 큰 특이값: 출력에 큰 영향을 미치는 중요한 방향
    # - 작은 특이값: 영향이 미미한 방향 -> 가지치기 대상!
    U, S, Vh = torch.linalg.svd(W)
    
    print(f"Original Singular Values: {S}")
    print(f"  - σ₁ (Major Axis): {S[0]:.4f}")
    print(f"  - σ₂ (Minor Axis): {S[1]:.4f}")
    print(f"  - Ratio (σ₂/σ₁): {(S[1]/S[0]).item():.2%}")
    # 예: tensor([3.85..., 1.15...]) -> 큰 축과 작은 축의 비율 확인
    # 비율이 작을수록 가지치기 시 정보 손실이 적음

    # ============================================================
    # 5. Low-Rank Approximation (Rank-1 근사) - 가지치기 시뮬레이션
    # ============================================================
    # 가장 작은 특이값을 0으로 만들어 행렬의 랭크를 감소시킴
    # 이는 네트워크 가지치기의 핵심 아이디어:
    # "중요하지 않은 연결(파라미터)을 제거하여 모델을 압축"
    # 
    # Rank 2 -> Rank 1 변환:
    # - 원래: 2개의 독립적인 출력 차원
    # - 가지치기 후: 1개의 출력 차원으로 압축 (1차원 직선)
    # 
    # 실제 신경망에서는:
    # - Rank 1000 -> Rank 100 등으로 압축
    # - 파라미터 수: d_out × d_in -> k(d_out + d_in) (k: 유지할 랭크)
    # - 예: 1000×1000 = 1M 파라미터 -> 100(1000+1000) = 200K (80% 압축)
    S_pruned = S.clone()
    S_pruned[-1] = 0  # 가장 작은 특이값 제거 (Min Singular Value -> 0)
    
    # 근사된 행렬 W_approx 재구성
    # W_approx = U @ diag(S_pruned) @ Vh
    # 이 행렬은 원본 W의 "가지치기된 버전"
    W_approx = U @ torch.diag(S_pruned) @ Vh
    
    # 정보 손실 정량화
    info_retained = (S_pruned**2).sum() / (S**2).sum()
    print(f"\nInformation Retained: {info_retained.item():.2%}")
    print(f"Information Lost: {(1-info_retained).item():.2%}")

    # ============================================================
    # 6. 가지치기된 모델의 출력 계산
    # ============================================================
    # 압축된 가중치 행렬 W_approx를 사용한 순전파
    # 단위 원이 직선(Line)으로 변형됨 -> 1차원으로 정보 손실
    output_pruned = input_points @ W_approx.T

    # ============================================================
    # 7. 시각화: 가지치기의 기하학적 효과 비교
    # ============================================================
    plt.figure(figsize=(15, 5))

    # Plot 1: 입력 공간 (단위 원)
    # 신경망의 다양한 입력을 대표하는 단위 벡터들
    plt.subplot(1, 3, 1)
    plt.scatter(input_points[:, 0], input_points[:, 1], c=theta, cmap='hsv', s=10)
    plt.title("1. Input Space (Unit Circle)\n모든 방향의 입력 벡터", fontsize=11)
    plt.xlabel("x₁", fontsize=10)
    plt.ylabel("x₂", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Plot 2: 원본 FC Layer 출력 (가지치기 전)
    # Full Rank 행렬: 2차원 정보 모두 보존
    plt.subplot(1, 3, 2)
    plt.scatter(output_original[:, 0], output_original[:, 1], c=theta, cmap='hsv', s=10)
    plt.title(f"2. Original Output (Ellipse)\nFull Rank=2 | Info={100:.1f}%", fontsize=11)
    plt.xlabel("y₁", fontsize=10)
    plt.ylabel("y₂", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # 주축(Principal Axes) 표시
    axis_major = U[:, 0] * S[0]
    axis_minor = U[:, 1] * S[1]
    plt.arrow(0, 0, axis_major[0], axis_major[1], head_width=0.3, 
              head_length=0.2, fc='darkblue', ec='darkblue', linewidth=2, 
              label=f'Major Axis (σ₁={S[0]:.2f})')
    plt.arrow(0, 0, axis_minor[0], axis_minor[1], head_width=0.2, 
              head_length=0.1, fc='darkred', ec='darkred', linewidth=2, 
              label=f'Minor Axis (σ₂={S[1]:.2f})')
    plt.legend(fontsize=8, loc='upper right')

    # Plot 3: 가지치기된 FC Layer 출력 (Rank-1)
    # 작은 특이값 제거 -> 정보가 1차원으로 압축됨
    plt.subplot(1, 3, 3)
    plt.scatter(output_pruned[:, 0], output_pruned[:, 1], c=theta, cmap='hsv', s=10)
    info_pct = info_retained.item() * 100
    plt.title(f"3. Pruned Output (Line)\nRank=1 | Info={info_pct:.1f}%", fontsize=11)
    plt.xlabel("y₁", fontsize=10)
    plt.ylabel("y₂", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # 제거된 축 설명
    plt.text(0, -2.5, "⚠ Minor Axis Collapsed (정보 손실)", 
             ha='center', color='red', fontsize=10, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("SVD 기반 네트워크 가지치기: 기하학적 해석", 
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('svd_pruning_visualization.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("✅ Visualization saved to svd_pruning_visualization.png")
    print("="*60)
    # plt.show()


if __name__ == "__main__":
    print("="*60)
    print("🚀 SVD 기반 네트워크 가지치기 시각화 시작")
    print("="*60)
    visualize_svd_pruning()
    print("\n💡 더 자세한 설명은 NETWORK_PRUNING_GUIDE.md를 참고하세요!")
    print("="*60)