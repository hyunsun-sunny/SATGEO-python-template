import argparse
from utils.cfg import Cfg
import torch
from pyproj import Proj, Transformer
import pyproj

#from app.models.{모델_py_파일} import {모델_클래스}
# from app.service.{서비스_py_파일} import {서비스_함수}
from models.models import Encoder, Decoder, Seq2Seq
import numpy as np

from utils.data_preprocessing import TestLoader
import datetime as dt
import pandas as pd
import time
## TODO: 관련 DB 모델 및 서비스 import 
# from  app.config.settings import settings

# from app.service import ship_prediction_route_service 

# class convertAxis():
#     def __init__(self):
#         self.projection_4326 = pyproj.CRS("EPSG:4326")
#         self.projection_3857 = pyproj.CRS("EPSG:3857")
#         self.transformer_4326_to_3857 = pyproj.Transformer.from_crs(self.projection_4326, self.projection_3857, always_xy=True)
#         self.transformer_3857_to_4326 = pyproj.Transformer.from_crs(self.projection_3857, self.projection_4326, always_xy=True)

#     def epsg4326TO3857(self, lon_data, lat_data):
#         # 좌표 변환: 4326 -> 3857
#         transformed_coords = [self.transformer_4326_to_3857.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
#         lat_3857, lon_3857 = zip(*transformed_coords)
#         return list(lon_3857), list(lat_3857)

#     def epsg3857T4326(self, lon_data, lat_data):
#         # 좌표 변환: 3857 -> 4326
#         transformed_coords = [self.transformer_3857_to_4326.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
#         lat_4326, lon_4326 = zip(*transformed_coords)

#         return list(lon_4326), list(lat_4326)


# def load_model():
#     torch.cuda.empty_cache()

#     checkpoint_path = f'{Cfg.model_path}/{Cfg.model_name}'
#     output_size = Cfg.output_size

#     model = Seq2Seq(input_size=Cfg.input_size, hidden_size=Cfg.hidden_size, output_size=output_size).to(Cfg.device)
#     checkpoint = torch.load(checkpoint_path, map_location=Cfg.device)
#     state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
#     model.load_state_dict(state_dict)
#     return model

# def Inference(model, source_seq):
    
#     print('Inference')
#     """
#     Inference with the Seq2Seq model.

#     Args:
#         model (nn.Module): 학습된 Seq2Seq 모델.
#         source_seq (np.array): 입력 시퀀스, shape [batch_size, source_length, input_size].
#         decoder_length (int): 예측하고자 하는 시퀀스 길이.

#     Returns:
#         np.array: 예측 결과, shape [batch_size, decoder_length, output_size].
#     """
#     device = next(model.parameters()).device
#     source_seq = torch.tensor(source_seq, dtype=torch.float32).to(device)
#     print('source_seq: ', source_seq.shape)
#     model.eval()
#     with torch.no_grad():
#         encoder_inputs = source_seq[:, :Cfg.encoder_length, :]
#         decoder_input = source_seq[:, Cfg.encoder_length - 1:Cfg.encoder_length, :]
        
#         # Decoder 첫 입력을 설정
#         predictions = model.inference(encoder_inputs, decoder_input, Cfg.decoder_length,Cfg.output_size)
#         print('predictions: ', predictions.shape)

#     return predictions.cpu().numpy()

# class Normalize():
#     def __init__(self):
#         self.min_values = Cfg.min_values
#         self.max_values = Cfg.max_values
#         self.date_value = self.max_values[0]
#         self.denominator = self.max_values - self.min_values

#     def normalize_data(self, data):
#         normalized_data = (data[:,:,1:5]-self.min_values[1:]) /self.denominator[1:]
#         data[:,:,1:5] = normalized_data
#         data[:,:,0] /= self.date_value
#         return data

#     def rescale_data(self, data):
#         data[:, 0] *= self.date_value
#         rescaled_data = data[:, 1:3] * self.denominator[1:3] + self.min_values[1:3]
#         data[:, 1:] = rescaled_data
#         return data


# def set_dataset(data):
#     # print('set_dataset', data)
#     seq_length = Cfg.encoder_length + Cfg.decoder_length
#     test_loader = TestLoader()
#     test_loader.loadTestTrajectory(data, seq_length)
#     if len(test_loader.trajectory)<1:
#         print('trajectory length is too short. shorter than 1')
#         del test_loader
#         return None
#     # normalize data
#     norm = Normalize()
#     test_loader.trajectory = norm.normalize_data(test_loader.trajectory)

#     seq_encoder = test_loader.trajectory
#     source_seq = seq_encoder[:,:Cfg.encoder_length, :5]
#     source_values = seq_encoder[:,:Cfg.encoder_length, 5:]

#     return seq_encoder, source_values
    

# def save_predictions(pred_values, source_values):
#     time_source = source_values[:, 0].tolist()
#     lng_source = source_values[:, 1].tolist()
#     lat_source = source_values[:, 2].tolist()

#     delta_time, delta_lng, delta_lat = pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]

#     time_pred, lng_pred, lat_pred = [], [], []

#     lng, lat = lng_source[-1], lat_source[-1]
#     time_i = time_source[-1]

#     for d_t, dlng, dlat in zip(delta_time, delta_lng, delta_lat):
#         time_i += d_t
#         lng += dlng
#         lat += dlat
#         time_pred.append(time_i)
#         lng_pred.append(lng)
#         lat_pred.append(lat)


#     time_source_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_source]
#     time_pred_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_pred]

#     # conver 3857 to 4326
#     axis = convertAxis()
#     lat_pred, lng_pred = axis.epsg3857T4326(lng_pred, lat_pred)
#     lat_source, lng_source = axis.epsg3857T4326(lng_source, lat_source)

#     source_data = {
#         'datetime': time_source_utc, 'lon': lng_source, 'lat': lat_source
#     }
#     pred_data = {
#         'datetime': time_pred_utc, 'lon': lng_pred, 'lat': lat_pred
#     }

#     return source_data, pred_data
    

# # def predict_routes_ais(db: Session, target_time: pd.datetime, mmsi: str):
# def predict_routes_ais(target_time: pd.datetime, mmsi: str):
#     # data = ship_prediction_route_service.get_ship_data(db, mmsi, target_time, previous_time)

#     ######################## test data ########################
#     files = ['SAIS01_20240101_000000__20240101_020000.csv', 'SAIS01_20240101_040000__20240101_060000.csv']
#             #  'SAIS01_20240101_060000__20240101_080000.csv', 'SAIS01_20240101_100000__20240101_120000.csv',
#             #  'SAIS01_20240101_120000__20240101_140000.csv', 'SAIS01_20240101_140000__20240101_160000.csv', 'SAIS01_20240101_160000__20240101_180000.csv',
#             #  'SAIS01_20240101_180000__20240101_200000.csv', 'SAIS01_20240101_220000__20240102_000000.csv']
#     file_path = 'D:/AIS_2024/'
#     chunk_size = 5000  # Adjust the chunk size based on your memory capacity
#     dataframes = []

#     for file in files:
#         for chunk in pd.read_csv(file_path + file, chunksize=chunk_size):
#             chunk = chunk[chunk['SOG'] >=10]
#             chunk['datetime'] = pd.to_datetime(chunk['Date']+ ' ' + chunk['Time'])
#             dataframes.append(chunk)
#             del chunk

#     data = pd.concat(dataframes, ignore_index=True)
#     # data = data[data['MMSI'] == '200660358']
#     # data = pd.read_csv('D:/AIS_2024/cluster_2597_351096000.csv')
    
    
#     grouped_data = data.groupby('MMSI')
#     # breakpoint()
    
#     mmsi = grouped_data.groups.keys()
#     for key, group in grouped_data:
#         print('key: ', key)
#         # print('group: ', group)
#         if len(group) >=30:
#             data = group
#             break
#     print(data)
#     source_length = Cfg.encoder_length+Cfg.decoder_length
#     model = load_model()
#     print('set data shap:',data.shape)
#     seq_encoder, source_values = set_dataset(data)
#     del data
#     ######################## test data ########################

#     pred_seq = Inference(model, seq_encoder[:,:source_length, :5])
#     pred_value = pred_seq[0]
#     norm = Normalize()
#     pred_value = norm.rescale_data(pred_value)

#     source_data, predict_data = save_predictions(pred_value, source_values)
#     print('source_data: ', source_data)
#     print('predic_data: ', predict_data)
#     save_data = {
#         'mmis': mmsi,
#         'lon': predict_data['lon'],
#         'lat': predict_data['lat'],
#         'datetime': predict_data['datetime']
#     }
#     # ship_prediction_route_service.save_ship_prediction_route(db, save_data)

    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mmsi", type=str, default='00000')
#     parser.add_argument("--target_time", type=str, default='2024-01-01 06:00:00')
#     args = parser.parse_args()
    
#     predict_routes_ais(mmsi=args.mmsi, target_time=args.target_time)

class ConvertAxis:
    def __init__(self):
        self.projection_4326 = pyproj.CRS("EPSG:4326")
        self.projection_3857 = pyproj.CRS("EPSG:3857")
        self.transformer_4326_to_3857 = pyproj.Transformer.from_crs(
            self.projection_4326, self.projection_3857, always_xy=True
        )
        self.transformer_3857_to_4326 = pyproj.Transformer.from_crs(
            self.projection_3857, self.projection_4326, always_xy=True
        )

    def epsg4326_to_3857(self, lon_data, lat_data):
        transformed_coords = [self.transformer_4326_to_3857.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
        lat_3857, lon_3857 = zip(*transformed_coords)
        return list(lon_3857), list(lat_3857)

    def epsg3857_to_4326(self, lon_data, lat_data):
        transformed_coords = [self.transformer_3857_to_4326.transform(lng, lat) for lng, lat in zip(lon_data, lat_data)]
        lat_4326, lon_4326 = zip(*transformed_coords)
        return list(lon_4326), list(lat_4326)


def load_model() -> torch.nn.Module:
    torch.cuda.empty_cache()

    checkpoint_path = f'{Cfg.MODEL_PATH}/{Cfg.MODEL_NAME}'
    output_size = Cfg.OUTPUT_SIZE

    model = Seq2Seq(
        input_size=Cfg.INPUT_SIZE,
        hidden_size=Cfg.HIDDEN_SIZE,
        output_size=output_size
    ).to(Cfg.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=Cfg.DEVICE)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model


def inference(model: torch.nn.Module, source_seq: np.ndarray) -> np.ndarray:
    print('Inference')
    device = next(model.parameters()).device
    source_seq = torch.tensor(source_seq, dtype=torch.float32).to(device)
    print('source_seq: ', source_seq.shape)
    model.eval()

    with torch.no_grad():
        encoder_inputs = source_seq[:, :Cfg.ENCODER_LENGTH, :]
        decoder_input = source_seq[:, Cfg.ENCODER_LENGTH - 1:Cfg.ENCODER_LENGTH, :]
        predictions = model.inference(
            encoder_inputs, decoder_input, Cfg.DECODER_LENGTH, Cfg.OUTPUT_SIZE
        )
        print('predictions: ', predictions.shape)

    return predictions.cpu().numpy()


class Normalize:
    def __init__(self):
        self.min_values = Cfg.MIN_VALUES
        self.max_values = Cfg.MAX_VALUES
        self.date_value = self.max_values[0]
        self.denominator = self.max_values - self.min_values

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        normalized_data = (data[:, :, 1:5] - self.min_values[1:]) / self.denominator[1:]
        data[:, :, 1:5] = normalized_data
        data[:, :, 0] /= self.date_value
        return data

    def rescale_data(self, data: np.ndarray) -> np.ndarray:
        data[:, 0] *= self.date_value
        rescaled_data = data[:, 1:3] * self.denominator[1:3] + self.min_values[1:3]
        data[:, 1:] = rescaled_data
        return data


def set_dataset(data: pd.DataFrame):
    seq_length = Cfg.ENCODER_LENGTH + Cfg.DECODER_LENGTH
    test_loader = TestLoader()
    test_loader.load_test_trajectory(data, seq_length)
    
    if len(test_loader.trajectory) < 1:
        print('Trajectory length is too short. Less than 1.')
        del test_loader
        return None
    
    norm = Normalize()
    test_loader.trajectory = norm.normalize_data(test_loader.trajectory)

    seq_encoder = test_loader.trajectory
    source_seq = seq_encoder[:, :Cfg.ENCODER_LENGTH, :5]
    source_values = seq_encoder[:, :Cfg.ENCODER_LENGTH, 5:]

    return seq_encoder, source_values


def save_predictions(pred_values: np.ndarray, source_values: np.ndarray):
    time_source = source_values[:, 0].tolist()
    lng_source = source_values[:, 1].tolist()
    lat_source = source_values[:, 2].tolist()

    delta_time, delta_lng, delta_lat = pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]
    time_pred, lng_pred, lat_pred = [], [], []

    lng, lat = lng_source[-1], lat_source[-1]
    time_i = time_source[-1]

    for d_t, dlng, dlat in zip(delta_time, delta_lng, delta_lat):
        time_i += d_t
        lng += dlng
        lat += dlat
        time_pred.append(time_i)
        lng_pred.append(lng)
        lat_pred.append(lat)

    time_source_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_source]
    time_pred_utc = [dt.datetime.fromtimestamp(ts / 1000.0) for ts in time_pred]

    axis = ConvertAxis()
    lat_pred, lng_pred = axis.epsg3857_to_4326(lng_pred, lat_pred)
    lat_source, lng_source = axis.epsg3857_to_4326(lng_source, lat_source)

    source_data = {'datetime': time_source_utc, 'lon': lng_source, 'lat': lat_source}
    pred_data = {'datetime': time_pred_utc, 'lon': lng_pred, 'lat': lat_pred}

    return source_data, pred_data

# def predict_routes_ais(db: Session,current_time:pd.Timestamp, target_time: pd.datetime, mmsi: str):
def predict_routes_ais(current_time:pd.Timestamp,target_time: pd.Timestamp, mmsi: str):
    start_time = time.time()
    
    # data = ship_prediction_route_service.get_ship_data(db, mmsi, target_time)
    ######################## test data ########################
    files = ['SAIS01_20240101_000000__20240101_020000.csv', 'SAIS01_20240101_040000__20240101_060000.csv']
    file_path = 'D:/AIS_2024/'
    chunk_size = 5000

    dataframes = []
    for file in files:
        for chunk in pd.read_csv(file_path + file, chunksize=chunk_size):
            chunk = chunk[chunk['SOG'] >= 10]
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'])
            dataframes.append(chunk)
            del chunk

    data = pd.concat(dataframes, ignore_index=True)
    grouped_data = data.groupby('MMSI')

    for key, group in grouped_data:
        if len(group) >= 30:
            data = group
            break
    ######################## test data ########################

    source_length = Cfg.ENCODER_LENGTH + Cfg.DECODER_LENGTH
    model = load_model()

    seq_encoder, source_values = set_dataset(data)
    # pred_seq = inference(model, seq_encoder[:, :source_length, :5])
    pred_seq = inference(model, seq_encoder[0:1, :source_length, :5])
    pred_value = Normalize().rescale_data(pred_seq[0])

    source_data, predict_data = save_predictions(pred_value, source_values[0])
    print('source_data: ', source_data)
    print('predict_data: ', predict_data)
    finish_time = time.time()
    if not (len(predict_data['lon']) == len(predict_data['lat']) == len(predict_data['datetime'])):
        raise ValueError("lon, lat, datetime 리스트의 길이가 같아야 합니다.")

    predict_route = [
        {"lon": predict_data['lon'][i], "lat": predict_data['lat'][i], "datetime": predict_data['datetime'][i]}
        for i in range(len(predict_data['lon']))
    ]
    print('predict_route: ', predict_route)
    save_data = {
        'mmsi': mmsi,
        'request_time': current_time,
        'target_time': target_time,
        'target_lon': source_data['lon'][-1], # 추후에 시간 맞춰 수정
        'target_lat': source_data['lat'][-1], # 추후에 시간 맞춰 수정
        'arrive_lon': predict_data['lon'][-1], 
        'arrive_lat': predict_data['lat'][-1],
        'require_time': start_time - finish_time, # 추후에 시간 맞춰 수정
        'distance': 0.0, # 추후에 산정하여 추가
        'predict_route': predict_route,
        'predict_circle': 3.14  # 예시 값, 확률반경 알고리즘 추후 추가

    }

    # ship_prediction_route_service.save_ship_prediction_route(db, save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmsi", type=str, default='00000')
    parser.add_argument("--target_time", type=str, default='2024-01-01 06:00:00')
    parser.add_argument("--current_time", type=str, default='2024-01-11 06:00:00')
    args = parser.parse_args()

    predict_routes_ais(current_time=pd.to_datetime(args.current_time),target_time=pd.to_datetime(args.target_time), mmsi=args.mmsi)

'''

'ship_id': save_data['mmsi'],  # 선박 ID
'request_time': save_data['predict_time'],  # 예측 시간
'prediction_route_type': 'example_type',  # 예시 값, 필요에 따라 수정

'standard_prediction_time': save_data['target_time'],  # 예측 기준 시간
'start_longitude': save_data['target_lon'],  # 선택한 선박 경도
'start_latitude': save_data['target_lat'],  # 선택한 선박 위도
'arrival_longitude': save_data['arrive_lon'],  # 마지막 시간 경로 (필요하면 계산)
'arrival_latitude': save_data['arrive_lat'],  # 도착 위도 (필요하면 계산)
'rp_type': 1,  # 예시 값, 필요에 따라 수정
'rp_requirement_second': save_data['require_time'],  # 필요에 따라 수정
'route_distance': save_data['distance'],  # 필요에 따라 수정
'route_requirement_second': 0.0,  # MTN에서 사용. 
'route': save_data['predict_route'],  # 예측 경로
'route_geom': None,  # 필요하면 Geometry 데이터 삽입
'predict_circle': 3.14  # 예시 값, 확률반경 알고리즘 추후 추가

'''