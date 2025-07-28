import pandas as pd
import numpy as np
import re
import nycflights13 as flights

# 1. 데이터 불러오기
print("=== 1. 데이터 불러오기 ===")
df_flights = flights.flights.copy()
df_planes = flights.planes.copy()

print(f"flights 데이터: {df_flights.shape}")
print(f"planes 데이터: {df_planes.shape}")

# 2. flights 데이터의 결측치 처리 (모두 결측이면 0으로 채우기)
print("\n=== 2. flights 데이터 결측치 처리 ===")
time_delay_cols = ['dep_time', 'arr_time', 'dep_delay', 'arr_delay', 'air_time']

print("처리 전 결측치 수:")
for col in time_delay_cols:
    print(f"{col}: {df_flights[col].isna().sum()}")

# 모든 컬럼이 결측인 행 찾기
all_missing_mask = df_flights[time_delay_cols].isna().all(axis=1)
print(f"\n모든 시간/지연 컬럼이 결측인 행의 수: {all_missing_mask.sum()}")

# 모든 컬럼이 결측인 행만 0으로 채우기
df_flights.loc[all_missing_mask, time_delay_cols] = 0

print("\n처리 후 결측치 수:")
for col in time_delay_cols:
    print(f"{col}: {df_flights[col].isna().sum()}")

# 3. tailnum 전처리 (양쪽 데이터프레임)
print("\n=== 3. tailnum 전처리 ===")
for df in [df_flights, df_planes]:
    df['tailnum_clean'] = df['tailnum'].astype(str).str.strip().str.upper()

print(f"flights tailnum 결측치: {df_flights['tailnum_clean'].isna().sum()}")
print(f"planes tailnum 결측치: {df_planes['tailnum_clean'].isna().sum()}")

# 4. 데이터 병합
print("\n=== 4. 데이터 병합 ===")
df = pd.merge(df_flights, df_planes, on='tailnum_clean', how='left', suffixes=('', '_plane'))
print(f"병합 후 데이터 형태: {df.shape}")

# planes 테이블의 year 컬럼이 year_plane으로 변경되었는지 확인
if 'year_plane' not in df.columns and 'year_y' in df.columns:
    df.rename(columns={'year_y': 'year_plane'}, inplace=True)
elif 'year_plane' not in df.columns and 'year' in df_planes.columns:
    # suffixes가 제대로 적용되지 않은 경우 수동으로 처리
    planes_rename = df_planes.rename(columns={'year': 'year_plane'})
    df = pd.merge(df_flights, planes_rename, on='tailnum_clean', how='left')

print("병합 후 year 관련 컬럼:")
year_cols = [col for col in df.columns if 'year' in col.lower()]
print(year_cols)

# 5. tailnum에서 숫자와 접미사 추출
print("\n=== 5. tailnum 파싱 (숫자 + 접미사 추출) ===")
def parse_tailnum(tail):
    """tailnum에서 숫자와 접미사를 분리"""
    if not isinstance(tail, str) or not tail.startswith('N'):
        return np.nan, np.nan
    
    tail = tail[1:]  # 'N' 제거
    match = re.match(r"(\d+)([A-Z]*)$", tail)
    
    if match:
        num = int(match.group(1))
        suffix = match.group(2) if match.group(2) else np.nan
        return num, suffix
    return np.nan, np.nan

# 숫자와 접미사 추출
parsed_tailnum = df['tailnum_clean'].apply(parse_tailnum)
df['tailnum_num'] = parsed_tailnum.apply(lambda x: x[0])
df['tail_suffix'] = parsed_tailnum.apply(lambda x: x[1])

print(f"tailnum_num 추출 성공: {df['tailnum_num'].notna().sum()}")
print(f"tail_suffix 추출 성공: {df['tail_suffix'].notna().sum()}")

# 6. model 결측치 채우기 (접미사-번호 기준 가장 가까운 모델)
print("\n=== 6. model 결측치 채우기 ===")
print(f"model 결측치 (처리 전): {df['model'].isna().sum()}")

# 모델이 있는 데이터만 참조용으로 사용
ref_model = df.loc[df['model'].notna(), ['tailnum_num', 'tail_suffix', 'model']].copy()

# 접미사가 없는 경우를 'NO_SUFFIX'로 처리
ref_model['tail_suffix'] = ref_model['tail_suffix'].fillna('NO_SUFFIX')

# 접미사별로 번호/모델 딕셔너리 생성 (빠른 검색용)
suffix_dict = {}
for suffix, group in ref_model.groupby('tail_suffix'):
    suffix_dict[suffix] = group[['tailnum_num', 'model']].values

print(f"접미사 종류: {list(suffix_dict.keys())}")
print(f"각 접미사별 데이터 개수: {[len(arr) for arr in suffix_dict.values()]}")

def find_nearest_model(row):
    """가장 가까운 번호의 모델 찾기"""
    if pd.notna(row['model']):
        return row['model']
    
    num = row['tailnum_num']
    suffix = row['tail_suffix'] if pd.notna(row['tail_suffix']) else 'NO_SUFFIX'
    
    # 숫자가 없으면 처리 불가
    if pd.isna(num):
        return np.nan
    
    # 1순위: 같은 접미사에서 찾기
    if suffix in suffix_dict and len(suffix_dict[suffix]) > 0:
        arr = suffix_dict[suffix]
        closest_idx = np.abs(arr[:, 0] - num).argmin()
        return arr[closest_idx, 1]
    
    # 2순위: 접미사가 없는 경우 전체에서 가장 가까운 번호 찾기
    if suffix == 'NO_SUFFIX':
        all_data = np.vstack(list(suffix_dict.values()))
        if len(all_data) > 0:
            closest_idx = np.abs(all_data[:, 0] - num).argmin()
            return all_data[closest_idx, 1]
    
    return np.nan

df['model'] = df.apply(find_nearest_model, axis=1)
print(f"model 결측치 (처리 후): {df['model'].isna().sum()}")

# 7. year 결측치 채우기 (모델+접미사 기준, 앞뒤 평균)
print("\n=== 7. year 결측치 채우기 ===")
print(f"year 결측치 (처리 전): {df['year_plane'].isna().sum()}")

# year가 있는 데이터만 참조용으로 사용
year_ref = df.loc[df['year_plane'].notna(), ['model', 'tail_suffix', 'tailnum_num', 'year_plane']].copy()
year_ref['tail_suffix'] = year_ref['tail_suffix'].fillna('NO_SUFFIX')

def interpolate_year(row):
    """앞뒤 번호의 년도 평균으로 보간"""
    if pd.notna(row['year_plane']):
        return row['year_plane']
    
    model = row['model']
    suffix = row['tail_suffix'] if pd.notna(row['tail_suffix']) else 'NO_SUFFIX'
    
    # 같은 모델, 같은 접미사 데이터 필터링
    condition = (
        (year_ref['model'] == model) &
        (year_ref['tail_suffix'] == suffix)
    )
    subset = year_ref[condition].sort_values('tailnum_num')
    
    if subset.empty or pd.isna(row['tailnum_num']):
        return np.nan
    
    target_num = row['tailnum_num']
    
    # 현재 번호보다 작은 번호 중 가장 큰 것
    before = subset[subset['tailnum_num'] < target_num]
    # 현재 번호보다 큰 번호 중 가장 작은 것
    after = subset[subset['tailnum_num'] > target_num]
    
    # 앞뒤 모두 있는 경우: 평균값
    if not before.empty and not after.empty:
        before_year = before['year_plane'].iloc[-1]
        after_year = after['year_plane'].iloc[0]
        return int(round(np.mean([before_year, after_year])))
    
    # 뒤만 있는 경우 (앞이 없음): 뒤 값 사용
    elif before.empty and not after.empty:
        return int(after['year_plane'].iloc[0])
    
    # 앞만 있는 경우 (뒤가 없음): 앞 값과 2013년의 평균
    elif not before.empty and after.empty:
        before_year = before['year_plane'].iloc[-1]
        return int(round(np.mean([before_year, 2013])))
    
    # 앞뒤 모두 없는 경우
    else:
        return np.nan

df['year_plane'] = df.apply(interpolate_year, axis=1)
print(f"year 결측치 (처리 후): {df['year_plane'].isna().sum()}")

# 8. 결과 확인
print("\n=== 8. 최종 결과 확인 ===")
print("주요 컬럼 결측치 현황:")
key_cols = ['tailnum', 'model', 'year_plane', 'type', 'manufacturer', 'seats']
for col in key_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        print(f"{col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")

print("\n샘플 데이터 (처리 결과):")
sample_cols = ['tailnum', 'tailnum_num', 'tail_suffix', 'model', 'year_plane', 'type', 'manufacturer']
available_cols = [col for col in sample_cols if col in df.columns]
print(df[available_cols].head(10))

# year 컬럼 정보 추가 확인
print(f"\nyear_plane 데이터 타입: {df['year_plane'].dtype}")
print(f"year_plane 값 범위: {df['year_plane'].min()} ~ {df['year_plane'].max()}")