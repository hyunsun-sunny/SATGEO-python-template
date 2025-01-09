# # -*- coding: utf-8 -*-
# """
# @Time          : 2024/12/18 00:00
# @Author        : Shinhye Han
# @File          : dataset.py
# @Noice         : 
# @Description   : Dataset definitions for ship classification tasks, including platform information.
# @How to use    : Import the `ShipClassificationDataset` class to load and preprocess datasets.

# @Modificattion :
#     @Author    :
#     @Time      :
#     @Detail    :
# """


# from pyproj import Proj, Transformer
# import pyproj

# import numpy as np
# # import Model
# import csv
# import time
# import pandas as pd
# import os
# import datetime as dt
# from scipy import interpolate
# from joblib import load
# from scipy.interpolate import PchipInterpolator

# import json
# import math


# def epsg4326TO3857(ais_data):
#     # EPSG:4326 (WGS 84) 좌표계와 EPSG:3857 (Web Mercator) 좌표계 설정
#     projection_4326 = pyproj.CRS("EPSG:4326")
#     projection_3857 = pyproj.CRS("EPSG:3857")

#     # 변환을 위한 pyproj 변환 객체 생성
#     transformer_4326_to_3857 = pyproj.Transformer.from_crs(projection_4326, projection_3857, always_xy=True)

#     # 좌표 변환: 4326 -> 3857
#     ais_data['lon'], ais_data['lat'] = zip(*ais_data.apply(lambda row: transformer_4326_to_3857.transform(row['lon'], row['lat']), axis=1))

#     return ais_data


# class TestLoader():
#     def __init__(self):
#         super().__init__()
#         self.trajectory_dict = {}
#         self.trajectory = []

#     def datetime_to_datenum(self, dtime):
#         mdn = dtime + dt.timedelta(days = 366)
#         frac_seconds = (dtime-dt.datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
#         frac_microseconds = dtime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
#         return mdn.toordinal() + frac_seconds + frac_microseconds    

#     def filter_continuous_data(self,df, max_gap_minutes=120, min_duration_hours=3):
#         df['datetime'] = pd.to_datetime(df['datetime'])
#         df = df.sort_values('datetime').reset_index(drop=True)
        
#         # Calculate time differences between consecutive rows
#         df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60.0  # in minutes
        
#         # Find indices where the gap is larger than max_gap_minutes
#         large_gaps = df[df['time_diff'] > max_gap_minutes].index
        
#         # Initialize start and end indices for the continuous segment
#         start_idx = 0
#         end_idx = len(df) - 1
        
#         # Iterate over large gaps to find a suitable continuous segment
#         for gap_idx in large_gaps:
#             if (df.loc[gap_idx - 1, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 >= min_duration_hours:
#                 end_idx = gap_idx - 1
#                 break
#             start_idx = gap_idx
        
#         # Ensure the segment is at least min_duration_hours long
#         if (df.loc[end_idx, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 < min_duration_hours:
#             raise ValueError("No continuous segment found with the required duration and gap constraints.")
        
#         return df.loc[start_idx:end_idx].reset_index(drop=True)


#     def process_csv(self, load_df):
#         df = load_df
#         print('load_df: ', load_df)
#         df = self.data_interpolation(df)
#         if df is None or df.empty:
#             return
        
#         df['Lon'] = df['lon'].round(5)
#         df['Lat'] = df['lat'].round(5)
#         df['COG'] = df['cog'].round(3)
#         df['SOG'] = df['sog'].round(3)
#         df['vx'] = df['vx'].round(3)
#         df['vy'] = df['vy'].round(3)

#         df['time_diff'] = [pd.Timestamp(it) for it in df['datetime']]
#         df['delta_time'] = df['time_diff'].diff().dt.total_seconds() * 1000
#         df['delta_lng'] = df['Lon'].diff()
#         df['delta_lat'] = df['Lat'].diff()
#         df['delta_cog'] = df['COG'].diff()
#         df['delta_sog'] = df['SOG'].diff()
       
#         df['time(ms)'] = df['datetime'].apply(lambda x: x.timestamp() * 1000)
#         # 첫 번째 행의 차이는 계산할 수 없으므로 0으로 채우기
#         df.at[0, 'delta_time'] = 0
#         df.at[0, 'delta_lng'] = 0
#         df.at[0, 'delta_lat'] = 0
#         df.at[0, 'delta_cog'] = 0
#         df.at[0, 'delta_sog'] = 0


#         # 필요한 컬럼만 선택하여 numpy array로 변환
#         point = df[['delta_time', 'delta_lng', 'delta_lat', 'vx','vy','time(ms)','Lon','Lat']].to_numpy()
#         point = point[1:]
#         return np.array(point)

    
#     def data_interpolation(self, df):
#         if df.empty:
#             print("Warning: DataFrame is empty!")
#             return None
#         df['datetime'] = pd.to_datetime(df['datetime'])

#         df['cog'] = df['cog'] % 360
#         df = df.dropna(subset=['datetime','lon', 'lat', 'cog', 'sog'])
#         df = df.drop_duplicates(subset=['datetime'])
#         if df.empty:
#             print("Warning: DataFrame is empty!")
#             return None
#         df.sort_values(['datetime'], inplace=True)
#         df = df[['datetime','lon', 'lat', 'cog', 'sog']]
#         time_interval = 10
#         df['sog'] *= 0.51444444


#         # COG, SOG -> vx, vy
#         # * 는 행렬 요소별 곱셈(matlab .*), @는 행렬 곱셈
#         vx = df['sog'].values * np.sin(list(map(lambda x: math.radians(x), df['cog'].values))) #python has no attribute 'sind'
#         vy = df['sog'].values * np.cos(list(map(lambda x: math.radians(x), df['cog'].values))) #python has no attribute 'cosd'
#         df.insert(3, 'vx', vx)
#         df.insert(4, 'vy', vy)    

#         # For Interpolation
#         df.set_index('datetime', inplace=True)
#         columns_to_interpolate = ['lon', 'lat','cog','sog', 'vx', 'vy']
#         interpolated_data = {}
#         df.index[0].replace(second=0, microsecond=0)
        
#         start_t = df.index[0]
#         finish_t= df.index[-1]
#         int_period = np.arange(start_t, finish_t, dt.timedelta(minutes=time_interval))
#         if int_period.size == 0:
#             print('error1')
#             return None

#         temp = list(map(lambda x: self.datetime_to_datenum(pd.to_datetime(x)), int_period))
#         min_x = min(temp)
#         temp = [max(val, min_x) for val in temp]
        
#         x = list(map(self.datetime_to_datenum, df.index))
#         # 중복된 x 값 제거
#         unique_x, unique_indices = np.unique(x, return_index=True)
        
#         df = df.iloc[unique_indices]
#         x = unique_x
#         print('error2')

#         for column in columns_to_interpolate:
#             f = PchipInterpolator(x, df[column], extrapolate=True)
#             # f = interpolate.interp1d(x, df[column], kind='quadratic', fill_value="extrapolate") # linear: 선형 보간, nearest: 가장 가까운 데이터 포인트 사용, zero: 0차 보간, slinear: 선형 보간의 스플라인 버전, quadratic, cubic: 2차, 3차 보간
#             interpolated_data[column] = f(temp)

#         interpolated_df = pd.DataFrame(interpolated_data)
#         interpolated_df.insert(0, 'datetime', int_period)
#         interpolated_df['datetime'] = pd.to_datetime(interpolated_df['datetime'])
#         interpolated_df = interpolated_df.drop_duplicates(subset=['datetime'])

#         # Now, interpolated_data is a dictionary where the keys are the column names
#         # and the values are the interpolated data for that column.
#         print('interpolated_df: ', interpolated_df)

#         return interpolated_df

    
#     def loadTestTrajectory(self, data, seq_length):
#         """Load trajectory data for testing. 

#         trajectory.shape = [N, 7]
#         The trajectories are converted to np.array([N, 7]), normalized and stored in self.trajectory. 
#         (including lng, lat)

#         Args:
#             file_name (string): file name of the csv.
#         """ 
#         load_df = data
#         load_df.columns = load_df.columns.str.lower()
#         print('len(load_df)', len(load_df))
#         if len(load_df) <5:
#             return
 
#         load_df = epsg4326TO3857(load_df)
#         load_df['datetime'] = pd.to_datetime(load_df['datetime'])
#         load_df = load_df.sort_values(by='datetime')
#         if len(load_df)<5:
#             return
#         start_i = 0 # 처음 index값
#         i=0
#         point = self.process_csv(load_df[start_i:i])

#         while i != len(load_df)-1:
#             if load_df['datetime'][i+1] - load_df['datetime'][i] > pd.Timedelta(hours=3):
#                 point = self.process_csv(load_df[start_i:i])
#                 for i in range(len(point) - seq_length):
#                     self.trajectory.append(point[i:i+seq_length])
#                 start_i = i+1
            
#             if i+1 == len(load_df)-1:
#                 point = self.process_csv(load_df[start_i:])
#                 for i in range(len(point) - seq_length):
#                     self.trajectory.append(point[i:i+seq_length].astype('float32'))
#                 break
#             i+=1
        
#         self.trajectory = np.array(self.trajectory)
       

# -*- coding: utf-8 -*-
"""
@Time          : 2024/12/18 00:00
@Author        : Shinhye Han
@File          : ship_classification_dataset.py
@Notice        : 
@Description   : Dataset definitions for ship classification tasks, including platform information.
@How to use    : Import the `ShipClassificationDataset` class to load and preprocess datasets.

@Modification :
    @Author    :
    @Time      :
    @Detail    :
"""

from pyproj import CRS, Transformer
from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd
import datetime as dt
import math


def epsg4326_to_3857(ais_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert coordinates from EPSG:4326 (WGS 84) to EPSG:3857 (Web Mercator).
    """
    projection_4326 = CRS("EPSG:4326")
    projection_3857 = CRS("EPSG:3857")
    transformer = Transformer.from_crs(projection_4326, projection_3857, always_xy=True)

    ais_data['lon'], ais_data['lat'] = zip(*ais_data.apply(
        lambda row: transformer.transform(row['lon'], row['lat']), axis=1
    ))
    return ais_data


class TestLoader:
    """
    Loader for testing ship trajectory data. Provides methods for interpolation and trajectory preparation.
    """

    def __init__(self):
        self.trajectory_dict = {}
        self.trajectory = []

    @staticmethod
    def datetime_to_datenum(dtime: dt.datetime) -> float:
        """
        Convert a datetime object to a MATLAB datenum format.
        """
        mdn = dtime + dt.timedelta(days=366)
        frac_seconds = (dtime - dt.datetime(dtime.year, dtime.month, dtime.day)).seconds / 86400.0
        frac_microseconds = dtime.microsecond / 86400.0 / 1e6
        return mdn.toordinal() + frac_seconds + frac_microseconds

    @staticmethod
    def filter_continuous_data(df: pd.DataFrame, max_gap_minutes: int = 120, min_duration_hours: int = 3) -> pd.DataFrame:
        """
        Filter continuous segments of data based on time gaps and minimum duration.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60.0
        large_gaps = df[df['time_diff'] > max_gap_minutes].index

        start_idx = 0
        end_idx = len(df) - 1

        for gap_idx in large_gaps:
            if (df.loc[gap_idx - 1, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 >= min_duration_hours:
                end_idx = gap_idx - 1
                break
            start_idx = gap_idx

        if (df.loc[end_idx, 'datetime'] - df.loc[start_idx, 'datetime']).total_seconds() / 3600.0 < min_duration_hours:
            raise ValueError("No continuous segment found with the required duration and gap constraints.")

        return df.loc[start_idx:end_idx].reset_index(drop=True)

    def process_csv(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process the trajectory data by interpolating and preparing features.
        """
        df = self.data_interpolation(df)
        if df is None or df.empty:
            return np.array([])

        df['lon'] = df['lon'].round(5)
        df['lat'] = df['lat'].round(5)
        df['cog'] = df['cog'].round(3)
        df['sog'] = df['sog'].round(3)
        df['vx'] = df['vx'].round(3)
        df['vy'] = df['vy'].round(3)

        df['time_diff'] = pd.to_datetime(df['datetime'])
        df['delta_time'] = df['time_diff'].diff().dt.total_seconds() * 1000
        df['delta_lng'] = df['lon'].diff()
        df['delta_lat'] = df['lat'].diff()
        df['delta_cog'] = df['cog'].diff()
        df['delta_sog'] = df['sog'].diff()

        df['time_ms'] = df['datetime'].apply(lambda x: x.timestamp() * 1000)
        df.iloc[0, df.columns.get_loc('delta_time'):] = 0

        return df[['delta_time', 'delta_lng', 'delta_lat', 'vx', 'vy', 'time_ms', 'lon', 'lat']].iloc[1:].to_numpy()

    def data_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate the trajectory data for continuous time intervals.
        """
        if df.empty:
            print("Warning: DataFrame is empty!")
            return None

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.dropna(subset=['datetime', 'lon', 'lat', 'cog', 'sog']).drop_duplicates(subset=['datetime'])
        if df.empty:
            print("Warning: DataFrame is empty after cleaning!")
            return None

        df = df.sort_values('datetime')
        df['cog'] %= 360
        df['sog'] *= 0.51444444
        df['vx'] = df['sog'] * np.sin(np.radians(df['cog']))
        df['vy'] = df['sog'] * np.cos(np.radians(df['cog']))

        df.set_index('datetime', inplace=True)
        columns_to_interpolate = ['lon', 'lat', 'cog', 'sog', 'vx', 'vy']
        time_interval = dt.timedelta(minutes=10)

        start_time = df.index[0].replace(second=0, microsecond=0)
        end_time = df.index[-1]
        time_points = pd.date_range(start=start_time, end=end_time, freq=time_interval)

        x = list(map(self.datetime_to_datenum, df.index))
        unique_x, unique_indices = np.unique(x, return_index=True)
        df = df.iloc[unique_indices]
        interpolated_data = {column: PchipInterpolator(unique_x, df[column])(list(map(self.datetime_to_datenum, time_points))) for column in columns_to_interpolate}

        interpolated_df = pd.DataFrame(interpolated_data)
        interpolated_df.insert(0, 'datetime', time_points)
        return interpolated_df.drop_duplicates(subset=['datetime'])

    def load_test_trajectory(self, data: pd.DataFrame, seq_length: int):
        """
        Load trajectory data for testing, preparing sequences of the specified length.
        """
        data.columns = data.columns.str.lower()
        data = epsg4326_to_3857(data)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values(by='datetime')

        start_idx = 0
        for i in range(len(data) - 1):
            if data.iloc[i + 1]['datetime'] - data.iloc[i]['datetime'] > pd.Timedelta(hours=3):
                self._process_segment(data[start_idx:i], seq_length)
                start_idx = i + 1

        self._process_segment(data[start_idx:], seq_length)
        self.trajectory = np.array(self.trajectory)

    def _process_segment(self, segment: pd.DataFrame, seq_length: int):
        """
        Process a segment of trajectory data into sequences.
        """
        points = self.process_csv(segment)
        for i in range(len(points) - seq_length):
            self.trajectory.append(points[i:i + seq_length].astype('float32'))
