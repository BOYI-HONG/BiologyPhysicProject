# tracker.py
import cv2
import pandas as pd
import numpy as np
import trackpy as tp

def extract_positions(video_path,max_frames=-1):
    if(max_frames!=-1):
        print(f"正在讀取影片 (僅處理前 {max_frames} 影格):\n{video_path}")
    else:
        print(f"正在讀取影片 (處理全影格):\n{video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案，請檢查路徑。")
        return pd.DataFrame()

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    data = []
    frame_idx = 0
    
    while cap.isOpened():
        if(max_frames!=-1):
            if frame_idx >= max_frames: 
                print(f"已達到 {max_frames} 影格限制，停止提取。")
                break
            
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask = fgbg.apply(blurred)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 400:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    
                    body_orientation = np.nan
                    if len(cnt) >= 5:
                        try:
                            _, _, angle = cv2.fitEllipse(cnt)
                            body_orientation = np.radians(angle)
                        except: pass
                    
                    data.append({
                        'frame': frame_idx, 
                        'x': cx, 'y': cy, 
                        'area': area,
                        'body_angle': body_orientation
                    })
        
        if frame_idx % 100 == 0: print(f"已處理 {frame_idx} 幀...")
        frame_idx += 1
        
    cap.release()
    return pd.DataFrame(data)

def link_data(df):
    print("正在連結軌跡...")
    t = tp.link_df(df, search_range=20, memory=3)
    t_filtered = tp.filter_stubs(t, threshold=20)
    
    # 【關鍵修復】暴力解除 Pandas 的 Index/Column 歧義
    # 1. 抹除索引的名稱，避免 Pandas 把索引誤認為 'frame' 欄位
    t_filtered.index.name = None
    # 2. 將索引強制重設為純數字 (0, 1, 2...)
    t_filtered = t_filtered.reset_index(drop=True)
    
    print(f"有效軌跡數量: {t_filtered['particle'].nunique()}")
    return t_filtered


def calculate_movement_angles(df):
    print("正在計算運動相位角...")
    # 再次確認索引乾淨
    df.index.name = None
    df = df.reset_index(drop=True)
    
    # 現在可以安全地對 'particle' 和實體欄位 'frame' 進行排序了
    df = df.sort_values(by=['particle', 'frame']).copy()
    
    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])
    
    df['d_phi'] = df.groupby('particle')['move_angle'].diff()
    df['d_phi'] = (df['d_phi'] + np.pi) % (2 * np.pi) - np.pi
    
    return df
