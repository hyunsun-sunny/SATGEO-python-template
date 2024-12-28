# -*- coding: utf-8 -*-
'''
@Time          : 2024/12/18 00:00
@Author        : Shinhye Han
@File          : s05_classify_unidentified_ships.py
@Noice         : 
@Description   : Perform multi-class classification on unidentified ship chips from SAR.
@How to use    : python s05_classify_unidentified_ships.py --input_dir {image path} --meta_file {metafile path}

@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torchvision import transforms

from utils.datasets import ShipClsDataset
from models.misc import select_model
from utils.cfg import ALGORITHM_NAME

## TODO: 관련 DB 모델 및 서비스 import 
from app.config.settings import settings
from app.models. import 
from app.service import 


def classify_unidentified_ships(db: Session, satellite_sar_image_id: str) -> None:
    """
    Perform multi-class classification on ship images.

    Args:
        input_dir (str): Path to input images.
        output_dir (str): Path to save output CSV.
        meta_file (str): Path to meta information file.
        classes (List[str]): List of ship classes.
        img_size (int): Size to which images will be resized.
    """
    input_dir = settings.S05_INPUT_PATH
    output_dir = settings.S05_OUTPUT_PATH
    meta_file = settings.S05_META_FILE
    
    start_time = time.time()

    # Image Preprocessing
    img_transforms = transforms.Compose([
        transforms.Pad(padding=(img_size, img_size), fill=0),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        lambda x: (x > 1000) * 1000 + (x < 1000) * x,
        lambda x: 255 * (x - x.min()) / (x.max() - x.min()),
        lambda x: x / 255,
        lambda x: x.repeat(3, 1, 1),
    ])

    # Load Model and Dataset
    model = select_model(classes, meta_file)
    dataset = ShipClassificationDataset(input_dir, transform=img_transforms, classes=classes)

    ## TODO: DB 검색하는 서비스 호출 
    # ecmwf_collect_hist_service.get_ecmwf_collect_history(db, transaction_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    init_time = time.time()

    # Perform Classification
    predictions = []
    labels = []

    for img, label in iter(dataset):
        labels.append(label)
        img = img.to(device).unsqueeze(0)
        y_pred, _ = model(img)

        _, top_pred = y_pred.topk(2, 1)
        predictions.append(top_pred[0][0].detach().cpu())

    # Save Classification Results
    results_df = pd.DataFrame({
        'FileName': dataset.img_files,
        'TrueClass': [classes[label] for label in labels],
        'PredClass': [classes[pred] for pred in predictions],
    })

    ## TODO: 산출물을 DB 테이블 포맷으로 변경 -> 모델 활용
    for i, pred in enumerate(predictions):
        record = SarShipUnidentification(
                satellite_sar_image_id=satellite_sar_image_id,
                unidentification_ship_id=i,
                
            )
            bulk_data.append(record)

        results_df.to_csv(output_dir, index=False)

    ## TODO: DB 모델에 컬럼 삽입할 서비스 호출
    # ecmwf_collect_hist_service.update_ecmwf_collect_history(db, collect_history)

    print(f"Results saved to {output_dir}")
    print(f"Done. Total Time: {1E3 * (time.time() - start_time):.1f}ms")
    print(f"Initialization Time: {1E3 * (init_time - start_time):.1f}ms")

     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ALGORITHM_NAME)
    parser.add_argument('-i', "--input_dir", type=str, required=True, default="/platform/data/inputs/, help="Path to input images")
    parser.add_argument('-o', "--output_dir", type=str, required=True, default="/platform/data/outputs/predictions.csv", help="Path to save output CSV")
    parser.add_argument('-m', '--meta_file', type=str, required=True, help="Path to meta information file")
    parser.add_argument('-s', '--img_size', type=int, default=224, help="Image size for preprocessing")

    args = parser.parse_args()
    args.classes = ['Cargo', 'Fishing', 'Sailing', 'Tanker', 'TugTow'] 

    classify_unidentified_ships(**vars(args)
