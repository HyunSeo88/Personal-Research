# SAR SLC 기반 초해상화(SR) 설계 원칙 정리

## ✅ 핵심 목적

SAR 영상에서 InSAR 분석이 가능하려면 SR 모델이 **SLC(Single Look Complex)** 형식의 입력을 받아 **SLC를 출력**해야 한다. 이는 위상 정보를 유지하여 interferogram 생성을 가능하게 하며, 진폭 기반 향상(amplitude-only SR)과 구별된다.

##  설계 원리 검증 흐름

### 1. SLC 입력-출력 요건 정당성

- **SLC**: 복소값 영상 (I: In-phase, Q: Quadrature), 위상 φ = arctan(Q/I)
- InSAR은 φ의 차이를 기반으로 변위/고도를 계산함
- **진폭만 SR**할 경우 coherence 손실 → InSAR 오류 10\~30% 증가 가능
- **정량 지표**: coherence γ = |\<s1·conj(s2)>| / sqrt(<|s1|²><|s2|²>)

### 2. SLC-SR 구현 시 문제점

- Bicubic interpolation은 speckle 증폭 → Rayleigh 분포 노이즈로 위상 decorrelation
- ***DL 모델****은 phase equivariance 보장해야 함*
- *최근 연구: amplitude-only SR은 TerraSAR-X 기준 coherence 0.2–0.3 하락*
- ***물리 제약 도입****: radar equation 기반 →* 위상 unwrap 과정에서 모호성 조절 필요

##  SLC→SLC SR 모델 유형

### 1) 복소값 CNN (Complex-Valued CNN)

- **구성**: 복소 tensor 입력 → 복소 합성곱 + holomorphic 활성화 → 복소 SLC 출력
- **장점**: 위상 손실 최소화, coherence > 0.85
- **단점**: 연산량 높음, 과적합 가능
- **적합 예시**: 도시 침하, 고정밀 InSAR
- **프레임워크**: PyTorch `torch.complex`, UNet 구조 기반

### 2) GAN 기반 위상 제약 혼합 모델

- **구성**: generator는 진폭/위상 dual-branch, discriminator는 interferometric loss 포함
- **장점**: speckle 질감 보존, 현실적 표현력, coherence 0.1\~0.2 향상
- **단점**: mode collapse 가능성, 훈련 불안정성
- **적합 예시**: 다변화 지형 (예: 홍수 분석)

### 3) 물리 기반 반복 복원 (Physics-Informed Iterative SR)

- **구성**: HQS 기반 알고리즘 unroll → sparse 정규화 + radar prior
- **장점**: 위상 안정성, 해석 가능, NMSE < 0.05
- **단점**: 느린 수렴, 잡음 일반화 어려움
- **적합 예시**: 센서 파라미터가 고정된 Sentinel-1 InSAR 스택

## 🛠️ 실용적 실행 계획

- **데이터**: Sentinel-1 SLC (ESA Copernicus 무료)
- **프레임워크**: PyTorch + complextorch 활용
- **검증**: SR 전/후 interferogram 생성 → coherence histogram 비교



