from PIL import Image
from tqdm import tqdm

import pandas as pd
import os

def crop_and_resize_images(df, input_base_path, output_folder, crop_size=94, resize_size=188, stride=47):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    new_df_rows = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        img_id = row['img_id']
        relative_img_path = row['img_path'].replace('./', '')
        img_path = os.path.join(input_base_path, relative_img_path)
        image = Image.open(img_path)

        for i in range(0, image.width, stride):
            for j in range(0, image.height, stride):
                if i + crop_size <= image.width and j + crop_size <= image.height:
                    box = (i, j, i + crop_size, j + crop_size)
                    cropped_image = image.crop(box)
                    resized_image = cropped_image.resize((resize_size, resize_size), Image.ANTIALIAS)

                    # 새로운 이미지 파일 이름
                    new_img_id = f"{img_id}_{i}_{j}.jpg"
                    output_path = os.path.join(output_folder, new_img_id)
                    resized_image.save(output_path)

                    # 새 이미지에 대한 정보를 데이터프레임에 추가
                    new_row = row.copy()
                    new_row['img_id'] = new_img_id
                    new_row['img_path'] = output_path
                    new_df_rows.append(new_row)
    
    new_df = pd.DataFrame(new_df_rows)
    return new_df

train_df = pd.read_csv('data/train.csv')

# 입력 및 출력 폴더 경로 정의
input_base_path = 'data/train'  # 원본 이미지가 위치한 경로
output_folder = 'data/aug_train'  # 증강된 이미지를 저장할 경로

# 함수 호출
augmented_df = crop_and_resize_images(train_df, input_base_path, output_folder)

# 증강된 데이터셋을 확인
augmented_df.head()
augmented_csv_path = 'data/augmented_train.csv'
augmented_df.to_csv(augmented_csv_path, index=False)