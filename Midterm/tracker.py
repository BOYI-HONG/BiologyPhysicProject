import cv2
import pandas as pd
import numpy as np
import trackpy as tp
import os
import config  # <--- 匯入集中管理的設定檔

# ==========================================
# 1. 強化版影像預處理 (Otsu + Morphological Closing)
# ==========================================
def extract_positions(video_path, max_frames=-1):
    if max_frames != -1:
        print(f"正在使用 Otsu+形態學 提取座標 (前 {max_frames} 影格):\n{video_path}")
    else:
        print(f"正在使用 Otsu+形態學 提取座標 (全影格):\n{video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案，請檢查路徑。")
        return pd.DataFrame()

    data_frames = []
    frame_idx = 0
    
    # 定義形態學的 Kernel (5x5 橢圓)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    while cap.isOpened():
        if max_frames != -1 and frame_idx >= max_frames: 
            break
            
        ret, frame = cap.read()
        if not ret: break
        
        # --- 影像處理 Pipeline ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用 Otsu 尋找最佳閾值並二值化
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形態學閉運算 (填補失焦破洞)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取座標與面積
        frame_data = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 使用 config 裡面的面積限制
            if config.AREA_MIN < area < config.AREA_MAX:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    
                    # 恢復擬合身體角度
                    body_orientation = np.nan
                    if len(cnt) >= 5:
                        try:
                            _, _, angle = cv2.fitEllipse(cnt)
                            body_orientation = np.radians(angle)
                        except: pass
                    
                    frame_data.append({
                        'frame': frame_idx, 
                        'x': cx, 'y': cy, 
                        'area': area,
                        'body_angle': body_orientation
                    })
        
        if frame_data:
            data_frames.append(pd.DataFrame(frame_data))
            
        if frame_idx % 100 == 0: 
            print(f"已處理 {frame_idx} 幀...")
        frame_idx += 1
        
    cap.release()
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()

# ==========================================
# 2. 軌跡連結 (Trackpy)
# ==========================================
def link_data(df):
    print("正在連結軌跡...")
    # 使用 config 裡面的 Trackpy 參數
    t = tp.link_df(df, search_range=config.SEARCH_RANGE, memory=config.MEMORY)
    t_filtered = tp.filter_stubs(t, threshold=config.THRESHOLD)
    
    t_filtered.index.name = None
    t_filtered = t_filtered.reset_index(drop=True)
    
    print(f"有效軌跡數量: {t_filtered['particle'].nunique()}")
    return t_filtered

# ==========================================
# 3. 物理量計算 (相位角)
# ==========================================
def calculate_movement_angles(df):
    print("正在計算運動相位角...")
    df.index.name = None
    df = df.reset_index(drop=True)
    
    df = df.sort_values(by=['particle', 'frame']).copy()
    
    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])
    
    df['d_phi'] = df.groupby('particle')['move_angle'].diff()
    df['d_phi'] = (df['d_phi'] + np.pi) % (2 * np.pi) - np.pi
    
    return df