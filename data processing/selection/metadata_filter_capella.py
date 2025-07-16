#!/usr/bin/env python3
"""
Capella SAR 메타데이터 필터링 스크립트
- ECEF 좌표를 위경도로 변환
- 한국 지역 데이터 필터링
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF 좌표를 위경도로 변환 (WGS84)"""
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis
    f = 1/298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e2 = 1 - (b**2 / a**2)  # eccentricity squared
    
    # Calculate longitude
    lon = np.arctan2(y, x)
    
    # Iterative calculation for latitude and height
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(5):  # iterate to improve accuracy
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    
    # Convert to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    return lat_deg, lon_deg, h


def parse_capella_metadata(filepath: Path) -> Optional[Dict]:
    """Capella 메타데이터 파싱"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 기본 정보 추출
        collect = data.get('collect', {})
        
        # 파일명에서 극화 정보 추출
        filename = filepath.name
        if '_VV_' in filename:
            polarization = 'VV'
        elif '_HH_' in filename:
            polarization = 'HH'
        else:
            polarization = 'Unknown'
        
        # 모드 정보
        mode = collect.get('mode', '')
        
        # 시간 정보
        start_time = collect.get('start_timestamp', '')
        
        # 플랫폼
        platform = collect.get('platform', '')
        
        # 해상도 정보 (GEO 제품의 경우)
        image = collect.get('image', {})
        pixel_spacing_row = image.get('pixel_spacing_row', None)
        pixel_spacing_column = image.get('pixel_spacing_column', None)
        
        # 중심 좌표 계산 (state vectors의 중간 지점 사용)
        state_vectors = data.get('state', {}).get('state_vectors', [])
        if state_vectors:
            # 중간 지점의 state vector 선택
            mid_idx = len(state_vectors) // 2
            mid_vector = state_vectors[mid_idx]
            pos = mid_vector['position']
            
            # ECEF to 위경도 변환
            lat, lon, height = ecef_to_geodetic(pos[0], pos[1], pos[2])
        else:
            lat, lon = None, None
        
        # Radar 정보
        radar = data.get('radar', {})
        center_freq = radar.get('center_frequency', 0) / 1e9  # Hz to GHz
        
        # 입사각 계산 (간단한 추정)
        # 실제로는 더 복잡한 계산이 필요하지만, 여기서는 대략적인 값 사용
        # Capella는 일반적으로 20-55도 범위
        incidence_angle = 35.0  # 기본값
        
        return {
            'filename': filename,
            'filepath': str(filepath),
            'platform': platform,
            'mode': mode,
            'polarization': polarization,
            'collect_date': start_time[:10] if start_time else None,
            'center_lat': lat,
            'center_lon': lon,
            'pixel_spacing_row': pixel_spacing_row,
            'pixel_spacing_column': pixel_spacing_column,
            'azimuth_resolution_m': pixel_spacing_row,
            'range_resolution_m': pixel_spacing_column,
            'center_frequency_ghz': center_freq,
            'incidence_angle_deg': incidence_angle,
            'product_type': data.get('product_type', '')
        }
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def filter_capella_data(
    metadata_list: List[Dict],
    roi_center: Tuple[float, float] = (36.5, 127.5),
    roi_radius_deg: float = 2.0,
    modes: List[str] = ['spotlight', 'stripmap'],
    max_resolution_m: float = 1.0,
    polarizations: List[str] = ['VV', 'HH']
) -> List[Dict]:
    """필터링 조건 적용"""
    filtered = []
    
    for meta in metadata_list:
        if not meta:
            continue
            
        # 1. 지리적 위치 필터
        if meta['center_lat'] is None or meta['center_lon'] is None:
            continue
            
        lat_diff = abs(meta['center_lat'] - roi_center[0])
        lon_diff = abs(meta['center_lon'] - roi_center[1])
        if lat_diff > roi_radius_deg or lon_diff > roi_radius_deg:
            continue
        
        # 2. 모드 필터
        if meta['mode'].lower() not in [m.lower() for m in modes]:
            continue
        
        # 3. 해상도 필터 (GEO 제품만)
        if meta['product_type'] == 'GEO':
            if meta['azimuth_resolution_m'] and meta['azimuth_resolution_m'] > max_resolution_m:
                continue
        
        # 4. 극화 필터
        if meta['polarization'] not in polarizations:
            continue
        
        filtered.append(meta)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Capella SAR 메타데이터 필터링')
    parser.add_argument('--capella-dir', type=str, 
                       default='../../metadata/HR/capella_data',
                       help='Capella 메타데이터 디렉토리')
    parser.add_argument('--output', type=str,
                       default='capella_filtered_metadata.csv',
                       help='출력 CSV 파일')
    
    args = parser.parse_args()
    
    # Capella 메타데이터 파일 찾기
    print("Capella 메타데이터 파일 검색 중...")
    capella_files = list(Path(args.capella_dir).rglob('*extended.json'))
    print(f"발견된 파일 수: {len(capella_files)}")
    
    # 메타데이터 파싱
    print("메타데이터 파싱 중...")
    metadata_list = []
    for filepath in capella_files:
        meta = parse_capella_metadata(filepath)
        if meta:
            metadata_list.append(meta)
    
    print(f"파싱 성공: {len(metadata_list)}개")
    
    # 필터링
    print("\n필터링 적용 중...")
    print("- ROI: 한반도 중심 (36.5°N, 127.5°E) ± 2°")
    print("- 모드: spotlight, stripmap")
    print("- 최대 해상도: 1.0m")
    print("- 극화: VV, HH")
    
    filtered = filter_capella_data(metadata_list)
    print(f"\n필터링 결과: {len(filtered)}개")
    
    # 결과 저장
    if filtered:
        df = pd.DataFrame(filtered)
        df.to_csv(args.output, index=False)
        print(f"\n결과 저장: {args.output}")
        
        # 통계 출력
        print("\n=== 필터링된 데이터 통계 ===")
        print(f"플랫폼별 분포:")
        print(df['platform'].value_counts())
        print(f"\n모드별 분포:")
        print(df['mode'].value_counts())
        print(f"\n극화별 분포:")
        print(df['polarization'].value_counts())
        
        # 지역별 분포
        print(f"\n지역별 분포:")
        for _, row in df.iterrows():
            print(f"  - {row['platform']}: ({row['center_lat']:.2f}°N, {row['center_lon']:.2f}°E)")
    else:
        print("필터링 조건을 만족하는 데이터가 없습니다.")


if __name__ == "__main__":
    main() 