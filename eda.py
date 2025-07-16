import pandas as pd
import numpy as np
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

# flights	2013년 뉴욕에서 출발한 항공편 정보
# airlines	항공사 코드와 이름 - carrier
# airports	공항 코드, 이름, 위치 정보
# planes	항공기 관련 정보
# weather	기상 정보

# 예시: 항공편 데이터 확인
print(df_flights.head())

# info()로 요약정보 확인
df_flights.info()
df_airlines.info()
df_airports.info()
df_planes.info()
df_weather.info()

# 컬럼명 확인
df_flights.columns
df_airlines.columns
df_airports.columns
df_planes.columns
df_weather.columns

# 결측값 확인
df_flights.isna().sum()
df_airlines.isna().sum()
df_airports.isna().sum()
df_planes.isna().sum()
df_weather.isna().sum()

# 고유값 확인
df_flights.nunique()
df_airlines.nunique()
df_airports.nunique()
df_planes.nunique()
df_weather.nunique()

df_weather["origin"].unique()