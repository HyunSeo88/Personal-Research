#!/usr/bin/env python3
"""
통합 SAR 메타데이터 필터링 스크립트
- Umbra와 Capella 데이터 통합 분석
- 한국 및 주변 지역 데이터 선별
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF 좌표를 위경도로 변환"""
    a = 6378137.0
    f = 1/298.257223563
    b = a * (1 - f)
    e2 = 1 - (b**2 / a**2)
    
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    
    return np.degrees(lat), np.degrees(lon), h


def parse_umbra_metadata(filepath: Path) -> Optional[Dict]:
    """Umbra 메타데이터 파싱"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        collect = data.get('collects', [{}])[0]
        
        # 위치 정보
        center_point = collect.get('sceneCenterPointLla', {})
        coords = center_point.get('coordinates', [])
        if len(coords) < 2:
            return None
        lon, lat = coords[0], coords[1]
        
        # 다른 정보
        pol = collect.get('polarizations', [''])[0]
        
        return {
            'sensor': 'Umbra',
            'filename': filepath.name,
            'filepath': str(filepath),
            'satellite': data.get('umbraSatelliteName', ''),
            'mode': data.get('imagingMode', ''),
            'polarization': pol,
            'collect_date': collect.get('startAtUTC', '')[:10],
            'center_lat': lat,
            'center_lon': lon,
            'azimuth_resolution_m': collect.get('maxGroundResolution', {}).get('azimuthMeters', None),
            'range_resolution_m': collect.get('maxGroundResolution', {}).get('rangeMeters', None),
            'incidence_angle_deg': collect.get('angleIncidenceDegrees', None),
            'product_type': data.get('productSku', '')
        }
    except Exception:
        return None


def parse_capella_metadata(filepath: Path) -> Optional[Dict]:
    """Capella 메타데이터 파싱"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        collect = data.get('collect', {})
        
        # 파일명에서 극화 정보
        filename = filepath.name
        if '_VV_' in filename:
            polarization = 'VV'
        elif '_HH_' in filename:
            polarization = 'HH'
        else:
            polarization = 'Unknown'
        
        # state vectors에서 위치 계산
        state = collect.get('state', {})
        state_vectors = state.get('state_vectors', [])
        
        if not state_vectors:
            return None
            
        mid_vector = state_vectors[len(state_vectors) // 2]
        pos = mid_vector.get('position', [])
        
        if not pos:
            return None
            
        lat, lon, _ = ecef_to_geodetic(pos[0], pos[1], pos[2])
        
        # 해상도 정보
        image = collect.get('image', {})
        
        return {
            'sensor': 'Capella',
            'filename': filename,
            'filepath': str(filepath),
            'satellite': collect.get('platform', ''),
            'mode': collect.get('mode', ''),
            'polarization': polarization,
            'collect_date': collect.get('start_timestamp', '')[:10],
            'center_lat': lat,
            'center_lon': lon,
            'azimuth_resolution_m': image.get('pixel_spacing_row'),
            'range_resolution_m': image.get('pixel_spacing_column'),
            'incidence_angle_deg': 35.0,  # 기본값
            'product_type': data.get('product_type', '')
        }
    except Exception:
        return None


def main():
    print("통합 SAR 메타데이터 필터링")
    print("=" * 60)
    
    # 1. Umbra 데이터 처리
    print("\n1. Umbra 데이터 처리...")
    try:
        umbra_files = list(Path('../../metadata/HR/umbra_data').rglob('*METADATA.json'))
        print(f"   Umbra 파일: {len(umbra_files)}개")
        
        umbra_data = []
        for filepath in umbra_files:
            try:
                meta = parse_umbra_metadata(filepath)
                if meta:
                    umbra_data.append(meta)
            except Exception as e:
                print(f"   Umbra 파일 처리 오류: {filepath.name} - {e}")
                continue
    except Exception as e:
        print(f"   Umbra 디렉토리 접근 오류: {e}")
        umbra_data = []
    
    # 2. Capella 데이터 처리
    print("\n2. Capella 데이터 처리...")
    try:
        capella_files = list(Path('../../metadata/HR/capella_data').rglob('*extended.json'))
        print(f"   Capella 파일: {len(capella_files)}개")
        
        capella_data = []
        for i, filepath in enumerate(capella_files):
            try:
                if i % 500 == 0:
                    print(f"   진행 중... {i}/{len(capella_files)}")
                meta = parse_capella_metadata(filepath)
                if meta and meta['center_lat'] is not None:
                    capella_data.append(meta)
            except Exception as e:
                print(f"   Capella 파일 처리 오류: {filepath.name} - {e}")
                continue
    except Exception as e:
        print(f"   Capella 디렉토리 접근 오류: {e}")
        capella_data = []
    
    # 3. 데이터 통합
    print("\n3. 데이터 통합...")
    all_data = umbra_data + capella_data
    
    if not all_data:
        print("처리할 데이터가 없습니다.")
        return
        
    df_all = pd.DataFrame(all_data)
    
    # 4. 한국 지역 필터링
    # 한국 본토: 33-39°N, 125-130°E
    # 확장 지역: 30-42°N, 120-135°E (북한, 서해, 동해 포함)
    
    df_korea = df_all[
        (df_all['center_lat'] > 33) & (df_all['center_lat'] < 39) &
        (df_all['center_lon'] > 125) & (df_all['center_lon'] < 130)
    ]
    
    df_korea_extended = df_all[
        (df_all['center_lat'] > 30) & (df_all['center_lat'] < 42) &
        (df_all['center_lon'] > 120) & (df_all['center_lon'] < 135)
    ]
    
    # 5. 결과 출력
    print("\n=== 통합 결과 ===")
    print(f"전체 데이터: {len(df_all)}개")
    print(f"  - Umbra: {len(umbra_data)}개")
    print(f"  - Capella: {len(capella_data)}개")
    print(f"\n한국 본토 (33-39°N, 125-130°E): {len(df_korea)}개")
    print(f"한국 확장 지역 (30-42°N, 120-135°E): {len(df_korea_extended)}개")
    
    # 센서별 분포
    if not df_korea_extended.empty:
        print("\n=== 한국 확장 지역 센서별 분포 ===")
        print(df_korea_extended['sensor'].value_counts())
        
        print("\n=== 한국 확장 지역 상세 ===")
        for _, row in df_korea_extended.iterrows():
            print(f"\n{row['sensor']} - {row['filename'][:40]}...")
            print(f"  위치: {row['center_lat']:.2f}°N, {row['center_lon']:.2f}°E")
            print(f"  날짜: {row['collect_date']}")
            print(f"  해상도: {row['azimuth_resolution_m']:.2f}m")
    
    # 6. 결과 저장
    df_korea_extended.to_csv('korea_region_sar_metadata.csv', index=False)
    print(f"\n한국 지역 SAR 데이터가 'korea_region_sar_metadata.csv'에 저장되었습니다.")
    
    # 연구용 데이터 선별 (고해상도, 최신)
    df_research = df_korea_extended[
        (df_korea_extended['azimuth_resolution_m'] < 1.0) &
        (df_korea_extended['collect_date'] > '2023-01-01')
    ]
    
    if not df_research.empty:
        df_research.to_csv('korea_research_priority.csv', index=False)
        print(f"연구 우선순위 데이터 {len(df_research)}개가 'korea_research_priority.csv'에 저장되었습니다.")
    else:
        print("연구 우선순위 데이터가 없습니다.")


if __name__ == "__main__":
    main() 