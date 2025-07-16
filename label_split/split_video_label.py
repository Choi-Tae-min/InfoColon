import os
import cv2
import pandas as pd

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def process_csv_and_generate_frames(csv_file_path, video_file_path, output_base_folder):
    # CSV 파일 불러오기
    df = pd.read_csv(csv_file_path)
    
    # 비디오 ID 추출 (예: CNUH0001)
    video_id = os.path.splitext(os.path.basename(video_file_path))[0]
    
    # 비디오 파일 읽기
    cap = cv2.VideoCapture(video_file_path)
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if success:
            frame_count += 1
            frame_file_name = f"{frame_count:06d}.png"  # 예: frame000001.png
            
            # 해당 프레임의 라벨 및 폴더 가져오기
            row = df[df['Filename'] == frame_file_name]
            if not row.empty:
                labels = row['Label'].values[0].split(', ')
                folder_name = row['train_val_test'].values[0]
                
                # 각 라벨에 맞는 하위 폴더 생성 (train/레이블, val/레이블, test/레이블)
                for label in labels:
                    label_folder = os.path.join(output_base_folder, folder_name, label)
                    create_folder_if_not_exists(label_folder)
                    
                    # 프레임 저장
                    new_frame_name = f"cnuh_{video_id}_{label}_{frame_count:06d}.png"
                    frame_file_path = os.path.join(label_folder, new_frame_name)
                    cv2.imwrite(frame_file_path, frame)
                    print(f"Saved {frame_file_path}")
    
    cap.release()
    print(f"Total frames saved and renamed for {video_id}: {frame_count}")

def process_all_videos_and_csvs(video_folder, csv_folder, output_base_folder):
    # 비디오 파일 목록과 CSV 파일 목록을 가져오기
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith('.csv')])
    
    for video_file, csv_file in zip(video_files, csv_files):
        video_file_path = os.path.join(video_folder, video_file)
        csv_file_path = os.path.join(csv_folder, csv_file)
        
        print(f"Processing: {video_file} with {csv_file}")
        process_csv_and_generate_frames(csv_file_path, video_file_path, output_base_folder)

# 사용 예시
video_folder = 'E:/colon/label_total/vid/'  # 비디오 파일들이 저장된 폴더
csv_folder = 'E:/colon/label_total/excel/'  # CSV 파일들이 저장된 폴더
output_base_folder = 'E:/colon/label_total/output/'  # train, val, test 폴더의 상위 폴더 경로

process_all_videos_and_csvs(video_folder, csv_folder, output_base_folder)