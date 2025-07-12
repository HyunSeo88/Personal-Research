# 📡 Personal Research — SAR Blind Super-Resolution & Domain Generalization  
**Author:** _H L_ • **Timeline:** 2025‒2026 (주 10 h+)  

---

## 1. 연구 주제 한눈 요약
Sentinel-1 저해상(≈10 m) 이미지를 **공개 GF-3 1 m 영상** 등으로 교사 삼아,  
복소(Self-Attention) 기반 **Blind Super-Resolution** 모델을 설계하고,  
RADARSAT-2 · ICEYE · TerraSAR-X 같은 **미학습 센서**에서도 성능 저하 없이 적용되는  
**도메인 일반화(DG)** 절차까지 완성하는 것이 목표이다.

---

## 2. 최근 연구 흐름 
| 흐름 | 핵심 내용 | 한계 |
|------|----------|------|
|**CNN/GAN → Transformer·Diffusion**<br>(EAM ‘25, Freq-Sep Transformer ‘25)|복잡한 열화 추정을 자체 Attention이나 확산(DDPM)으로 대체|실수 이미지 위주, SAR Phase·Speckle 미반영|
|**Blind SR 확대**<br>(Prob. CycleGAN ‘23)|Degradation Net+SR Net로 HR 참조 없이 학습|센서 간 통계 편차 보정 부족|
|**Complex-Valued Network**<br>(CV-ViT ‘25)|위상(Coherence)까지 학습|학습 안정성·메모리 부담|
|**DG 연구 부상**<br>(MixStyle ‘24, GRL ‘23)|특징 통계 섞기·도메인 역전 학습|SAR 전용 벤치마크 미비|

---

## 3. 새롭게 탐구할 핵심 포인트 
1. **Curriculum Cascaded SR**  
   – 4× 두 단계를 순차 학습해 ×16까지 안정적으로 확대  
2. **Speckle-Aware Dual Loss**  
   – Phase-Cosine(위상 일관) + Speckle-KL(감마 분포 정합)로 물리 모델 반영  
3. **DG 모듈**  
   – MixStyle(Magnitude/Phase 분리 섞기), GRL 기반 도메인 판별기, Prototype Alignment 비교  
4. **경량화 & 단일 GPU 대응**  
   – 4070 Ti 환경에서 batch = 1, AMP, Patch-wise 학습 → 실험 재현성 보장  

---

## 4. 구체 연구 질문
1. 복소 Self-Attention이 실수 Transformer 대비 **CPSNR·V-STID**를 얼마나 개선할까?  
2. **Speckle-KL** vs. 기존 Charbonnier/Adversarial Loss: 노이즈 잔존량 차이는?  
3. 어떤 DG 기법이 미학습 센서(RADARSAT-2, ICEYE)에 가장 적은 성능 드롭을 보일까?  

---

## 5. 예상 로드맵 (초안)  
| Phase | 기간 | 산출물 |
|-------|------|--------|
|**Concept**|M0–M0.5|SAR 물리·스펙클 통계 개념 정리 노트|
|**Survey**|M0.5–M1|최신 Blind SR·DG 논문 리뷰|
|**DataPrep**|M1–M1.5|Sentinel-1 ↔ GF-3 정합·전처리 파이프라인, QGIS 스크립트|
|**Modeling**|M1.5–M3|C-BiT-Cascader 구현, Dual Loss 탑재|
|**Experiment**|M3–M4|×16 최종 모델 학습, ablation + 메트릭 로그|
|**DG**|M4–M5|MixStyle·GRL·Prototype 탐색, 타깃 센서 테스트|
|**WriteUp**|M5–M6|논문 초안 + 코드/DVC 리포 공개|

---

## 6. 데이터·도구 📦
- **데이터**: Sentinel-1 SLC, GF-3 Spotlight, 공개 RADARSAT-2/ICEYE 샘플  
- **프레임워크**: PyTorch ≥ 2.3 + Torch-complex, QGIS 3.36  
- **HW**: RTX 4070 Ti (12 GB) × 1, 4 TB SSD, Ubuntu 22.04  
- **버전관리**: Git + DVC, 모든 실험 seed 고정  

---

## 7. 기대 산출물
- ✏️ 학술 논문
- 🖥️ 공개 코드 저장소 
- 📊 DG 벤치마크 리포트(센서 4종, 메트릭 3종)  

---

## 8. 초기 참고문헌 (2023–2025)  
- Xie H. et al., “EAM: Enhancing Anything with Diffusion Transformers for Blind SR,” _arXiv_, 2025.  
- Zhai L. et al., “Degradation-Aware Frequency-Separated Transformer,” _LNCS_, 2025.  
- Zhang C. et al., “Blind SR for SAR with Speckle Noise based on Prob. CycleGAN,” _Remote Sensing_, 2023.  
- Luo Y. et al., “Complex-Valued Multiscale Attention ViT for PolSAR,” _Information Fusion_, 2025.  
- Peng Y. et al., “MixStyle for Domain Generalization,” _CVPR Workshop_, 2024.

---
