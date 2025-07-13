# 연구 제안서 v2.0 (하이브리드 설계 확정)

**제목**: 복소수 Transformer를 이용한 SAR 초해상화 모델의 센서 간 강인성 연구: C-BiT & A-BiT 하이브리드 설계를 통한 위상 정보 역할 증명

**연구자**: 임현서  

**기간**: 2025-08 ~ 

**최종 수정**: 2025년 7월 13일

---

## 1. 연구 개요 (Executive Summary)

기존 합성개구레이더(SAR) 딥러닝 연구는 대부분 위상(Phase) 정보를 소실한 진폭(Amplitude) 이미지를 사용함으로써, 센서의 물리적 특성을 온전히 활용하지 못하는 근본적 한계를 지닌다. 이는 모델이 특정 센서의 데이터 분포에 과적합되어, 다른 종류의 센서 데이터에 대해서는 성능이 급격히 저하되는 문제로 이어진다.

본 연구는 이러한 한계를 극복하기 위해 **"위상까지 학습한 복소 Transformer(C-BiT)는 진폭 전용 모델(A-BiT)보다 미학습 센서·밴드에서 성능 하락폭이 작다"**는 핵심 가설을 설정하고, 이를 '인과 → 현실' 두 단계로 엄밀하게 검증한다.

---

## 2. 연구 배경 및 핵심 가설

### 2.1 문제 정의: 진폭 정보의 한계와 도메인 종속성

대부분의 SAR 딥러닝 모델은 GRD(Ground Range Detected)와 같은 진폭 데이터만을 사용한다. 이는 구현이 용이하다는 장점이 있으나, 센서의 고유한 주파수(C-band, X-band 등), 해상도, 스펙클(Speckle) 통계에 모델이 종속되는 결과를 낳는다. 결과적으로, Sentinel-1(C-band)으로 학습한 모델은 TerraSAR-X(X-band) 데이터 앞에서 성능이 급격히 저하되어 실제 활용성이 떨어진다.

### 2.2 핵심 가설: 위상 정보는 일반화의 열쇠

SAR 원시 데이터(SLC)에 포함된 위상 정보는 단순 노이즈가 아닌, 지표면의 미세 구조와 산란 메커니즘을 담고 있는 핵심 정보다. 우리는 위상 정보가 센서의 종류에 비교적 덜 민감한, 더 근본적이고 '일반화된(generalized)' 물리적 특징을 담고 있을 것이라 가정한다.

> **핵심 가설**: 복소수 연산을 통해 위상 정보를 학습한 모델(C-BiT)은, 동일한 구조의 진폭 전용 모델(A-BiT)보다 이종(unseen) 센서 데이터에 대해 더 낮은 성능 하락률을 보일 것이다.

---

## 3. 제안 방법론: 2-Stage 하이브리드 프레임워크

|  Stage             |  목표                        |  데이터                                               |  구조                              |  평가지표                                 |
| ------------------ | -------------------------- | -------------------------------------------------- | -------------------------------- | ------------------------------------- |
| **0 (Controlled)** | 위상 유무 성능 차를 깨끗이 증명         | Capella SLC 0.5m → 합성 LR 10m                     | 샴 : C-BiT vs A-BiT (L1 손실)       | CPSNR, Drop Rate, V-STID              |
| **1 (Real)**       | 현실 Unpaired 환경에서 위상 효과 재검증 | Sentinel-1 (LR) ↔ Capella·Radarsat-2 (HR) Unpaired | CycleGAN 열화 + C-BiT/CVT Cascader | Zero-shot ΔCPSNR, Coherence Drop, DRI |

### 3.1 Stage-0: 통제된 환경에서의 인과관계 증명 (샴 구조)

**목표**: 위상 정보의 유무가 모델의 강인성에 미치는 순수한 영향을 다른 변수 없이 깨끗하게 측정.

```
Input LR/HR ┐           ┌─> C-BiT(복소)
            ├─ 동일 구조 ┤
            └─> A-BiT(실수)
```

**네트워크 구조**:
- 실험군 (C-BiT): 복소수 SLC 데이터를 입력받는 Complex Transformer 기반 생성자
- 대조군 (A-BiT): 진폭 데이터를 입력받는 실수 연산 Transformer. C-BiT와 파라미터 수를 동일하게 맞춘 등가 모델 (≈ 42M)
- 손실: L1 + Phase-Cos (복소), A-BiT는 Phase-Cos 생략

**데이터**: 완벽한 LR-HR 쌍으로 구성된 합성 데이터셋
- 고해상도 위성 데이터(Capella, KOMPSAT-5, TerraSAR-X)를 '정답 HR'로 설정
- 알려진 열화 함수(PSF 블러, 스펙클 추가 등)를 적용하여 '문제지 LR'을 생성

### 3.2 Stage-1: 실제 환경에서의 적용 가능성 검증 (CycleGAN 구조)

**목표**: Stage 0에서 증명된 위상 정보의 효과가, 짝이 없고(unpaired) 센서 종류도 다른 현실 데이터 환경에서도 유효함을 입증.

```
LR ──► G_LR→HR (CVT ×4→×8) ───► SR HR
 ▲                                 │
 │ Cycle(L_cyc)                    ▼
HR ◄── G_HR→LR (CVT Down) ◄────── HR
```

**네트워크 구조**:
- CycleGAN 프레임워크를 기반으로 하며, 두 개의 생성자(LR↔HR) 모두 C-BiT 아키텍처를 사용
- 복소 QKV : 12L, head 8, Window 7×7, Low-Rank 32
- GAN + Cycle + Phase-Cos + Speckle-KL + Domain-Adv

**데이터**: 실제 비대응(Unpaired) 데이터셋
- LR 도메인: 무료로 대량 확보 가능한 저해상도 위성 데이터 (Sentinel-1 C-band)
- HR 도메인: 확보 가능한 고해상도 위성 데이터 (Capella, KOMPSAT-5, TerraSAR-X 등 X-band)

---

## 4. 데이터 & 전처리

- **LR 도메인** – Sentinel-1 IW SLC 300 scene → 256² 패치 60k
- **HR 도메인** – Capella 40·Umbra 20·RADARSAT-2 20 scene → 1024² 패치 6k
- 궤도 보정 → 열 잡음 제거 → σ°(dB) 정규화 (-25~0 dB) → 패치 HDF5
- **합성 LR** : PSF 블러 + ×4 다운샘플 + Γ Speckle(L=1-4)

---

## 5. 손실 함수 가중치

\(\mathcal{L}_{tot}= \mathcal{L}_{adv}+10\,\mathcal{L}_{cyc}+5\,\mathcal{L}_{id}+1\,\mathcal{L}_{pc}+0.5\,\mathcal{L}_{kl}+\lambda_{dg}\,\mathcal{L}_{DA}\)

- **Stage-0** : λ_pc = 1, λ_kl = 0
- **Stage-1** : Epoch 20 후 λ_kl↑, λ_dg = 1
- **물리 기반 손실 함수**: Speckle KL-Divergence Loss를 함께 사용하여 모델이 물리적으로 타당한 결과를 생성하도록 유도

---

## 6. 평가 지표 및 검증 계획

본 연구는 '강인성'이라는 가설을 입증하기 위해 다각적인 평가 지표를 활용한다.

|  범주  |  지표                          |  설명                                               |  우선순위 |
| ---- | ---------------------------- | ------------------------------------------------- | ----- |
| 품질   | CPSNR (복소), SSIM             | 복소수 데이터의 전반적인 복원 품질 및 구조적 유사도 측정 (기본 베이스라인) | 핵심    |
| 물리   | V-STID (또는 ENL)             | 스펙클 통계 분포의 보존 정도를 측정 (물리적 타당성 검증)             | 핵심    |
| 물리   | Coherence Drop               | (데이터 확보 시) 위상 정보 보존 능력을 직접적으로 측정 (결정적 증거)    | 도전적   |
| 강인성  | Drop Rate %                  | (가설 검증 핵심 지표) 훈련 도메인 대비 테스트 도메인에서의 성능 하락률     | 핵심    |
| 강인성  | DRI (1-평균 Drop)             | 센서/밴드 전이 강인성 종합 지수                             | 핵심    |

**통계 검증**: seed 3 × 10 patch bootstrap → paired t-test (p < 0.05)

**검증 방법**:
- 학습된 두 모델을 처음 보는 테스트 데이터에 적용하여 성능 하락률(Drop Rate)을 직접 비교
- 학습에 사용되지 않은 제3의 센서(예: ALOS-2 L-band) 데이터에 대한 Zero-shot 초해상화를 수행하여 모델의 최종 일반화 성능을 평가

---

## 7. 일정 (6개월)

|  월   |  주요 작업                                          |
| ---- | ----------------------------------------------- |
| M0-1 | 데이터 크롤러·합성 LR 스크립트 완료 + Stage-0 학습              |
| M1-2 | Stage-0 분석·표 정리                                 |
| M2-4 | Stage-1 CycleGAN + CVT 학습 (AMP, batch 1)        |
| M4-5 | DG Ablation (MixStyle·GRL·Proto) + Zero-shot 평가 |
| M5-6 | 논문 작성, 코드·데이터 subset 공개                         |

**HW 요구사항**: RTX 4070 Ti 12GB (총 학습 14일 예상)

---

## 8. 기대 효과

### 8.1 학술적 기여
1. **Stage-0** : 위상 정보가 '강인성'에 기여함을 통제 환경에서 최초 정량 입증
2. **Stage-1** : Unpaired + 밴드 차 현실 조건에서도 효과 재현
3. SAR 딥러닝 분야에서 최초로 '위상 정보 활용'과 '센서 간 일반화 성능' 사이의 인과관계를 실험적으로 증명

### 8.2 실용적 가치
- 특정 센서에 종속되지 않는 강인한 모델을 개발함으로써, Sentinel-1, KOMPSAT, TerraSAR-X 등 가용한 다양한 위성 데이터를 통합적으로 활용
- 재난 감시, 국방 안보 등 실제 문제 해결 능력 향상
- 공개 레포·데이터 → SAR DG-SR 벤치마크 확대

---

## 9. 참고 핵심 문헌

- Zhang, C., et al. (2023). *Blind Super-Resolution for SAR Images with Speckle Noise Based on Probabilistic Cycle-GAN*. Remote Sensing.
- Wang, M., et al. (2025). *A Complex-valued SAR Foundation Model Based on Physically Inspired Representation Learning*. arXiv:2504.11999.
- Barrachina, J. A., et al. (2023). *Comparison Between Equivalent Architectures of Complex-valued and Real-valued Neural Networks*. Journal of Signal Processing Systems.
- Xie et al., *EAM Diffusion Transformer*, arXiv 2025.
- Luo et al., *Complex-ViT PolSAR*, Information Fusion 2025.
- Park, S. W., et al. (2022). *Deep Complex-valued Transformer for Audio Source Separation*. ICASSP. 