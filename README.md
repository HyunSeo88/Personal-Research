# 📡 SAR 초해상화 모델의 센서 간 강인성 연구 -보류
**복소수 Transformer를 이용한 위상 정보 기반 도메인 일반화**

**연구자:** 임현서 • **기간:** 2025-08 ~ 2026-02 (6개월)  
**HW 환경:** RTX 4070 Ti 12GB • **프레임워크:** PyTorch 2.3+ + Torch-complex

---

## 🎯 연구 목표 및 핵심 가설

### 핵심 가설
> **"위상까지 학습한 복소 Transformer(C-BiT)는 진폭 전용 모델(A-BiT)보다 미학습 센서·밴드에서 성능 하락폭이 작다"**

기존 SAR 딥러닝 모델이 특정 센서에 과적합되어 다른 센서 데이터에서 성능이 급격히 저하되는 문제를 해결하기 위해, **위상 정보**가 센서 간 일반화에 미치는 영향을 체계적으로 검증합니다.

### 연구 배경
- 기존 SAR 딥러닝: 진폭(GRD) 데이터만 사용 → 센서별 특성에 종속
- 위상 정보: 지표면 미세구조·산란 메커니즘 포함 → 센서 간 공통 물리적 특징
- 현실 문제: Sentinel-1 학습 모델이 TerraSAR-X에서 성능 급락

---

## 📋 연구 방법론: 2-Stage 하이브리드 프레임워크

### Stage-0: 통제된 환경에서의 인과관계 증명 (샴 구조)
**목표**: 위상 정보 유무가 모델 강인성에 미치는 순수한 영향 측정

```
합성 LR/HR ┐           ┌─> C-BiT (복소수 SLC 입력)
           ├─ 동일구조 ┤
           └─> A-BiT (진폭 데이터 입력)
```

- **데이터**: 고해상도 위성 데이터에서 합성한 완벽한 LR-HR 쌍
- **손실**: L1 + Phase-Cos (복소), A-BiT는 Phase-Cos 생략
- **평가**: CPSNR, V-STID, Drop Rate 직접 비교

### Stage-1: 실제 환경에서의 적용성 검증 (CycleGAN 구조)
**목표**: 현실 Unpaired 데이터에서도 위상 정보 효과 재검증

```
LR ──► G_LR→HR (CVT ×4→×8) ───► SR HR
 ▲                              │
 │ Cycle Loss                   ▼
HR ◄── G_HR→LR (CVT Down) ◄────── HR
```

- **데이터**: 실제 비대응(Unpaired) 센서 데이터
- **손실**: GAN + Cycle + Phase-Cos + Speckle-KL + Domain-Adv
- **평가**: Zero-shot ΔCPSNR, Coherence Drop, DRI

---

## 📊 데이터셋 구성

### 현재 데이터 상황
- **수집 상태**: 메타데이터 확보 완료 (전체 데이터 용량: 수십 TB)
- **저장 위치**: `metadata/HR/` (capella_data, umbra_data 디렉토리)
- **데이터 수집 전략**: 필요한 ROI 및 시기별 선별 다운로드 계획

### 데이터 구성 계획

#### 저해상도 (LR) 도메인
- **Sentinel-1 IW SLC** (C-band, ~20m 해상도)
- **수량**: 300 scene → 256² 패치 60k개
- **전처리**: 궤도 보정 → 열 잡음 제거 → σ°(dB) 정규화 (-25~0 dB)

#### 고해상도 (HR) 도메인  
- **Capella** (X-band, 0.5m 해상도)
  - 현재 상태: 메타데이터 확보, 선별 다운로드 예정
  - 목표: 40 scene → 1024² 패치 4k개
  
- **Umbra** (X-band, 0.25m 해상도)  
  - 현재 상태: 메타데이터 확보, 선별 다운로드 예정
  - 목표: 20 scene → 1024² 패치 2k개

#### 검증용 데이터 (Zero-shot 테스트)
- **RADARSAT-2** (C-band, ~3m 해상도): 20 scene
- **TerraSAR-X** (X-band, ~1m 해상도): 추가 확보 예정

### 합성 데이터 생성
- **방법**: PSF 블러 + ×4 다운샘플 + Γ Speckle(L=1-4)
- **용도**: Stage-0 통제 실험용 완벽한 LR-HR 쌍 생성

---

## 🏗️ 모델 아키텍처

### C-BiT (Complex Bi-level Transformer)
- **입력**: 복소수 SLC 데이터 (실수부 + 허수부)
- **구조**: 복소 QKV, 12 Layer, 8 Head, Window 7×7, Low-Rank 32
- **특징**: 위상 정보 학습을 위한 Complex Self-Attention

### A-BiT (Amplitude Bi-level Transformer)  
- **입력**: 진폭 데이터만 (|complex|)
- **구조**: C-BiT와 동일한 파라미터 수 (≈ 42M)
- **특징**: 실수 연산 기반 등가 대조군 모델

### 손실 함수
```
L_tot = L_adv + 10·L_cyc + 5·L_id + 1·L_pc + 0.5·L_kl + λ_dg·L_DA
```

- **L_pc**: Phase-Cosine Loss (위상 일관성)
- **L_kl**: Speckle KL-Divergence Loss (물리적 타당성)
- **L_DA**: Domain Adversarial Loss (도메인 일반화)

---

## 📈 평가 지표

### 성능 지표
| 지표 | 설명 | 우선순위 |
|------|------|----------|
| **CPSNR** | 복소수 데이터 복원 품질 | 핵심 |
| **V-STID** | 스펙클 통계 분포 보존도 | 핵심 |
| **Drop Rate** | 훈련→테스트 도메인 성능 하락률 | 핵심 |
| **DRI** | 센서/밴드 전이 강인성 지수 | 핵심 |
| **Coherence Drop** | 위상 정보 보존 능력 | 도전적 |

### 통계 검증
- **방법**: seed 3 × 10 patch bootstrap → paired t-test (p < 0.05)
- **검증**: 미학습 센서 데이터에 대한 Zero-shot 성능 평가

---

## 🛠️ 기술 스택

### 개발 환경
- **프레임워크**: PyTorch ≥ 2.3, Torch-complex
- **데이터 처리**: QGIS 3.36, GDAL, rasterio
- **하드웨어**: RTX 4070 Ti 12GB × 1, 4TB SSD
- **OS**: Ubuntu 22.04

### 최적화 기법
- **메모리**: AMP (Automatic Mixed Precision), batch=1
- **학습**: Patch-wise 학습, Gradient Accumulation
- **재현성**: 모든 실험 seed 고정, Git + DVC 버전관리

---

## 📅 연구 일정 (6개월)

| 월 | 주요 작업 | 상세 내용 |
|----|----------|-----------|
| **M0-1** | 데이터 준비 + Stage-0 | 선별 다운로드, 합성 LR 생성, 샴 구조 학습 |
| **M1-2** | Stage-0 분석 | C-BiT vs A-BiT 성능 비교, 통계 검증 |
| **M2-4** | Stage-1 구현 | CycleGAN + CVT 학습, 도메인 적응 |
| **M4-5** | 도메인 일반화 | MixStyle·GRL·Proto 비교, Zero-shot 평가 |
| **M5-6** | 논문 작성 | 결과 정리, 코드 공개, 벤치마크 리포트 |

**예상 GPU 사용량**: 총 14일 (4070 Ti 기준)

---

## 🎯 기대 효과

### 학술적 기여
1. **최초 정량 검증**: 위상 정보가 센서 간 강인성에 기여함을 통제된 환경에서 증명
2. **실용적 검증**: Unpaired + 다른 밴드 조건에서도 효과 재현
3. **벤치마크 구축**: SAR 도메인 일반화 평가 기준 제시

### 실용적 가치
- **다중 센서 활용**: Sentinel-1, Capella, Umbra 등 다양한 데이터 통합 활용
- **실전 적용**: 재난 감시, 국방 안보 등 실제 문제 해결
- **오픈소스 기여**: 공개 코드 저장소 + 벤치마크 데이터셋

---

## 📚 핵심 참고문헌

### 최신 연구 동향 (2023-2025)
- Wang, M., et al. (2025). *Complex-valued SAR Foundation Model*. arXiv:2504.11999
- Luo, Y., et al. (2025). *Complex-Valued Multiscale Attention ViT for PolSAR*. Information Fusion
- Zhang, C., et al. (2023). *Blind SR for SAR with Speckle Noise*. Remote Sensing
- Xie, H., et al. (2025). *EAM: Diffusion Transformers for Blind SR*. arXiv

### 도메인 일반화
- Barrachina, J. A., et al. (2023). *Complex-valued vs Real-valued Neural Networks*. JSPS
- Park, S. W., et al. (2022). *Deep Complex-valued Transformer*. ICASSP

---

## 📂 프로젝트 구조

```
Personal Research/
├── README.md                    # 이 파일
├── integrated_research_proposal.md  # 상세 연구 계획서
├── metadata/                    # 데이터 메타데이터
│   └── HR/
│       ├── capella_data/        # Capella 메타데이터
│       └── umbra_data/          # Umbra 메타데이터
├── project code/                # 구현 코드 (예정)
└── SAR study & Experiment/      # 실험 결과 (예정)
```

---

## 🚀 시작하기

1. **환경 설정**
   ```bash
   conda create -n sar-sr python=3.10
   conda activate sar-sr
   pip install torch torchvision torch-complex pandas numpy
   ```

2. **데이터 준비**
   - 메타데이터 기반 ROI 선별
   - 필요한 시기/지역 데이터만 선택적 다운로드
   - 전처리 파이프라인 실행

3. **모델 학습**
   - Stage-0: 통제된 환경에서 인과관계 증명
   - Stage-1: 실제 환경에서 적용성 검증

---

**🔗 관련 링크**: 
- [Capella Space](https://www.capellaspace.com/)
- [Umbra Lab](https://umbra.space/)
- [Sentinel-1 ESA](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)

---

## 🔧 메타데이터 필터링 도구

연구에 적합한 SAR 데이터를 선별하기 위한 Python 도구들이 `data processing/selection/` 디렉토리에 위치

### 통합 필터링 (`integrated_sar_metadata_filter.py`)
```bash
cd "data processing/selection"
python integrated_sar_metadata_filter.py
```

**주요 기능:**
- Umbra와 Capella 데이터 통합 분석
- 한국 지역 데이터 자동 필터링
- 연구 우선순위 데이터 선별

### Capella 전용 필터링 (`metadata_filter_capella.py`)
```bash
cd "data processing/selection"
python metadata_filter_capella.py --capella-dir ../../metadata/HR/capella_data
```

**필터링 조건:**
- ✅ **필수**: 지리적 위치, 해상도(<1m), 극화(VV/HH), 입사각(25-45°)
- ⚡ **선택**: 노이즈 레벨, 수집 날짜, 파일 크기

### 현재 데이터 현황 (2025년 1월 기준)
- **전체 데이터**: 200개 (한국 확장 지역)
  - **Umbra**: 188개 (우수한 커버리지)
  - **Capella**: 12개
- **연구 우선순위**: 189개 (해상도 <1m, 2023년 이후)
- **최고 해상도**: 0.047m (Capella, 북한 지역)
- **주요 타겟**: 평양, 인천공항, 서해 위성발사장 등

### 생성된 결과 파일
- `korea_region_sar_metadata.csv`: 전체 한국 지역 데이터 (200개)
- `korea_research_priority.csv`: 연구 우선순위 데이터 (189개)
- `capella_asia_full.csv`: Capella 아시아 전체 데이터
- `capella_korea_nearby_all.csv`: Capella 한국 근처 데이터
