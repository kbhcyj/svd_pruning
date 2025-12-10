# 네트워크 가지치기와 SVD: 이론과 실습

## 목차
1. [개요](#개요)
2. [네트워크 가지치기의 이해](#네트워크-가지치기의-이해)
3. [SVD 기반 가지치기의 원리](#svd-기반-가지치기의-원리)
4. [본 프로젝트와의 연관성](#본-프로젝트와의-연관성)
5. [다양한 가지치기 기법들](#다양한-가지치기-기법들)
6. [실제 응용 사례](#실제-응용-사례)
7. [참고 자료](#참고-자료)

---

## 개요

본 프로젝트는 딥러닝 모델 경량화의 핵심 기술인 **네트워크 가지치기(Network Pruning)**를 **특이값 분해(Singular Value Decomposition, SVD)**를 통해 기하학적으로 시각화합니다. 

이를 통해 다음을 이해할 수 있습니다:
- 가지치기가 모델의 표현력에 미치는 영향
- 가중치 행렬의 랭크(Rank) 감소가 의미하는 것
- 정보 손실과 모델 압축의 트레이드오프

---

## 네트워크 가지치기의 이해

### 1. 가지치기의 필요성

현대 딥러닝 모델들은 점점 커지고 있으며, 이는 다음과 같은 문제를 야기합니다:

- **메모리 부족**: 수십억 개의 파라미터를 저장하기 위한 메모리 필요
- **추론 속도 저하**: 많은 연산량으로 인한 실시간 응답 불가
- **에너지 소비**: 모바일/엣지 디바이스에서의 배터리 소모
- **배포 제약**: 제한된 하드웨어 환경에서의 실행 불가능

### 2. 가지치기의 핵심 아이디어

신경망의 **과잉 파라미터화(Over-parameterization)** 특성을 활용:
- 학습된 가중치 중 상당수는 출력에 미치는 영향이 미미함
- 이런 "불필요한" 가중치를 제거해도 성능 유지 가능
- **Lottery Ticket Hypothesis**: 큰 모델 내에 작은 서브네트워크가 존재

### 3. 가지치기의 분류

#### 구조적 가지치기 (Structured Pruning)
- **대상**: 뉴런, 채널, 필터 단위
- **장점**: 하드웨어 가속 용이, 실제 속도 향상
- **예시**: 전체 필터 제거, 채널 프루닝

#### 비구조적 가지치기 (Unstructured Pruning)
- **대상**: 개별 가중치 단위
- **장점**: 더 높은 압축률 달성 가능
- **단점**: 희소 행렬(Sparse Matrix) 연산 필요

---

## SVD 기반 가지치기의 원리

### 1. SVD 분해의 의미

임의의 행렬 $W \in \mathbb{R}^{m \times n}$은 다음과 같이 분해됩니다:

$$
W = U \Sigma V^T
$$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 출력 공간의 회전 (Left Singular Vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$: 각 축 방향의 스케일링 (Singular Values)
- $V^T \in \mathbb{R}^{n \times n}$: 입력 공간의 회전 (Right Singular Vectors)

### 2. 특이값의 해석

특이값 $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$ (단, $r = \text{rank}(W)$)은:
- **각 주성분(Principal Component)의 중요도**를 나타냄
- 큰 특이값: 해당 방향의 정보 보존량이 큼
- 작은 특이값: 해당 방향의 정보 기여도가 작음

### 3. Low-Rank Approximation

Rank-$k$ 근사 ($k < r$):

$$
W_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T
$$

이는 Frobenius Norm 기준으로 **최적의 Rank-$k$ 근사** (Eckart-Young-Mirsky Theorem)입니다:

$$
W_k = \arg\min_{\text{rank}(X) \leq k} \|W - X\|_F
$$

### 4. 정보 손실 정량화

근사 오차:

$$
\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$

상대적 정보 보존량:

$$
\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \times 100\%
$$

---

## 본 프로젝트와의 연관성

### 1. 코드 구조 분석

#### test.py의 주요 단계

```python
# Step 1: 2x2 가중치 행렬 정의
W = torch.tensor([[2.0, 1.0],
                  [1.5, 3.0]], dtype=torch.float32)
```

이는 완전 연결 계층(FC Layer)의 가중치 행렬을 시뮬레이션합니다.

```python
# Step 2: SVD 수행
U, S, Vh = torch.linalg.svd(W)
```

행렬을 성분별로 분해하여 각 축의 중요도(특이값)를 추출합니다.

```python
# Step 3: 가장 작은 특이값 제거 (가지치기)
S_pruned = S.clone()
S_pruned[-1] = 0  # Rank 2 -> Rank 1
```

이는 **구조적 가지치기의 수학적 모델**입니다.

```python
# Step 4: 근사된 행렬 재구성
W_approx = U @ torch.diag(S_pruned) @ Vh
```

가지치기된 모델의 가중치를 재구성합니다.

### 2. 기하학적 해석

#### 시각화의 의미

**Plot 1 - 입력 공간 (단위 원)**
- 모든 방향으로의 단위 입력 벡터를 표현
- 신경망의 입력 분포를 대변

**Plot 2 - 원본 출력 (타원)**
- 가중치 행렬 $W$가 입력을 어떻게 변형시키는지 표현
- 타원의 장축/단축은 두 특이값 $\sigma_1, \sigma_2$에 비례
- 2차원 정보가 모두 보존됨

**Plot 3 - 가지치기 후 출력 (직선)**
- 작은 특이값 제거로 인해 한 차원으로 정보가 압축됨
- **Rank Deficiency**: 더 이상 2차원 공간을 표현할 수 없음
- 주축(Major Axis) 방향의 정보만 남음

### 3. 실제 딥러닝과의 연결

#### FC Layer에서의 적용

일반적인 FC Layer: $y = Wx + b$

- 가중치 행렬 $W \in \mathbb{R}^{d_{out} \times d_{in}}$를 SVD로 분해
- Rank-$k$ 근사로 두 개의 작은 행렬로 대체:

$$
W \approx W_k = (U_k \sqrt{\Sigma_k})(\sqrt{\Sigma_k} V_k^T) = W_1 W_2
$$

여기서:
- $W_1 \in \mathbb{R}^{d_{out} \times k}$
- $W_2 \in \mathbb{R}^{k \times d_{in}}$

**파라미터 감소량**:
- 원본: $d_{out} \times d_{in}$
- 근사: $k(d_{out} + d_{in})$
- 압축률: $\frac{k(d_{out} + d_{in})}{d_{out} \times d_{in}}$

예를 들어, $d_{out} = d_{in} = 1000$, $k = 100$인 경우:
- 원본: 1,000,000 파라미터
- 근사: 200,000 파라미터
- **80% 압축 달성**

---

## 다양한 가지치기 기법들

### 1. Magnitude-based Pruning

**원리**: 가중치의 절댓값이 작은 것부터 제거

```python
# Pseudo-code
mask = torch.abs(W) > threshold
W_pruned = W * mask
```

**장점**: 단순하고 효과적
**단점**: 전역 최적이 아닐 수 있음

### 2. SVD-based Pruning (본 프로젝트)

**원리**: 작은 특이값에 대응하는 성분 제거

**장점**: 
- 수학적으로 최적 근사
- 정보 손실을 정량화 가능

**단점**: 
- SVD 계산 비용 (O(min(m²n, mn²)))
- 행렬 곱셈으로 분해 필요

### 3. Gradient-based Pruning

**원리**: 손실 함수에 미치는 영향(Hessian 근사)을 기준으로 제거

**대표 알고리즘**:
- Optimal Brain Damage (OBD)
- Optimal Brain Surgeon (OBS)

$$
\Delta L \approx \frac{1}{2} w_i^2 H_{ii}
$$

### 4. Lottery Ticket Hypothesis

**절차**:
1. 무작위 초기화로 큰 모델 학습
2. 작은 가중치들 제거 (마스크 생성)
3. **원래 초기값으로 재학습**

**발견**: "당첨 티켓" 서브네트워크는 처음부터 작게 학습해도 동일한 성능 달성

### 5. 비교표

| 방법 | 계산 복잡도 | 이론적 근거 | 구조 보존 | 재학습 필요 |
|------|------------|------------|----------|------------|
| Magnitude | 낮음 | 약함 | ❌ | ✅ |
| SVD | 중간 | 강함 | ✅ | ✅ |
| Gradient-based | 높음 | 강함 | ❌ | ✅ |
| Lottery Ticket | 매우 높음 | 중간 | ❌ | ✅✅ |

---

## 실제 응용 사례

### 1. Computer Vision

**ResNet SVD Pruning**:
- 각 Convolution Layer를 $k \times k \times c_{in} \times c_{out}$ 텐서로 간주
- 채널 차원으로 SVD 적용하여 Rank 감소
- ImageNet에서 2배 압축 + 1% 정확도 하락

### 2. Natural Language Processing

**BERT 압축**:
- 12층 Transformer의 각 Attention/FFN Layer에 SVD 적용
- Rank-64 근사로 모델 크기 40% 감소
- Fine-tuning 후 성능 회복

### 3. 실시간 추론

**MobileNet + Pruning**:
- Depthwise Separable Convolution에 채널 프루닝 적용
- 모바일 디바이스에서 30 FPS 달성
- 에너지 효율 60% 향상

---

## 실험 확장 제안

### 1. 다양한 Rank 실험

현재 코드를 수정하여 Rank-1이 아닌 다른 랭크 근사를 시도:

```python
# Rank-k 근사 함수
def rank_k_approximation(W, k):
    U, S, Vh = torch.linalg.svd(W)
    S_pruned = S.clone()
    S_pruned[k:] = 0  # k번째 이후 특이값 제거
    return U @ torch.diag(S_pruned) @ Vh
```

### 2. 정보 보존량 시각화

각 랭크에서의 정보 보존량 그래프:

```python
def plot_information_retention(W):
    U, S, Vh = torch.linalg.svd(W)
    total_energy = (S ** 2).sum()
    
    retention = []
    for k in range(1, len(S) + 1):
        energy_k = (S[:k] ** 2).sum()
        retention.append((energy_k / total_energy).item())
    
    plt.plot(range(1, len(S) + 1), retention, 'o-')
    plt.xlabel('Rank k')
    plt.ylabel('Information Retention (%)')
    plt.title('Energy Spectrum')
```

### 3. 고차원 확장

3D 또는 더 높은 차원으로 확장하여 실제 신경망과 유사한 환경 시뮬레이션:

```python
# 예: 100차원 -> 50차원 FC Layer
W_high_dim = torch.randn(50, 100)
U, S, Vh = torch.linalg.svd(W_high_dim)

# 상위 k개 특이값만 유지
k = 20  # 60% 압축
W_compressed = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
```

### 4. 실제 신경망 적용

PyTorch 모델의 FC Layer에 SVD 가지치기 적용:

```python
class PrunedLinear(nn.Module):
    def __init__(self, linear_layer, rank):
        super().__init__()
        W = linear_layer.weight.data
        U, S, Vh = torch.linalg.svd(W)
        
        # Rank-k 분해
        self.W1 = nn.Parameter(U[:, :rank] @ torch.diag(S[:rank].sqrt()))
        self.W2 = nn.Parameter(torch.diag(S[:rank].sqrt()) @ Vh[:rank, :])
        
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        out = x @ self.W2.T @ self.W1.T
        if self.bias is not None:
            out += self.bias
        return out
```

---

## 참고 자료

### 논문

1. **Low-Rank Approximation**
   - "Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation" (Denton et al., NIPS 2014)
   
2. **Magnitude Pruning**
   - "Learning both Weights and Connections for Efficient Neural Networks" (Han et al., NIPS 2015)
   
3. **Lottery Ticket Hypothesis**
   - "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, ICLR 2019)

4. **Optimal Brain Damage**
   - "Optimal Brain Damage" (LeCun et al., NIPS 1990)

### 도서

- "Deep Learning" (Goodfellow et al.) - Chapter 7: Regularization
- "Neural Network Compression" (Cheng et al.) - 전체 챕터

### 온라인 자료

- PyTorch Pruning Tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization

---

## 결론

본 프로젝트는 단순한 2차원 시각화를 통해 네트워크 가지치기의 **본질**을 보여줍니다:

✅ **가지치기 = 부분 공간(Subspace) 압축**  
✅ **특이값 = 정보 중요도의 정량적 지표**  
✅ **랭크 감소 = 표현력 손실의 기하학적 의미**

이러한 직관은 실제 고차원 신경망에서도 동일하게 적용되며, 모델 경량화를 위한 다양한 기법들의 이론적 토대가 됩니다.

**핵심 메시지**: 가지치기는 단순히 "연결 끊기"가 아니라, **정보 이론적으로 의미 있는 압축**입니다.

