# tracker.py
import cv2
import pandas as pd
import numpy as np
import trackpy as tp
import os

MOG_History = 500
MOG_VarTHR  = 5
ROI_Size    = [5,800]


def extract_positions(video_path,max_frames=-1):
    if(max_frames!=-1):
        print(f"正在讀取影片 (僅處理前 {max_frames} 影格):\n{video_path}")
    else:
        print(f"正在讀取影片 (處理全影格):\n{video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案，請檢查路徑。")
        return pd.DataFrame()

    fgbg = cv2.createBackgroundSubtractorMOG2(MOG_History, MOG_VarTHR, detectShadows=False)
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
            if ROI_Size[0]< area < ROI_Size[1]:
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
    t = tp.link_df(df, search_range=20, memory=10)
    t_filtered = tp.filter_stubs(t, threshold=20)
    
    # 【關鍵修復】暴力解除 Pandas 的 Index/Column 歧義
    # 1. 抹除索引的名稱，避免 Pandas 把索引誤認為 'frame' 欄位
    t_filtered.index.name = None
    # 2. 將索引強制重設為純數字 (0, 1, 2...)
    t_filtered = t_filtered.reset_index(drop=True)
    
    print(f"有效軌跡數量: {t_filtered['particle'].nunique()}")
    return t_filtered

def render_tracking_video(video_path, df, output_path, fps=30.0, tail_seconds=10):
    """
    讀取原始影片，疊加眼蟲輪廓與歷史軌跡，並輸出為新影片。
    
    參數:
    video_path (str): 原始影片路徑。
    df (pd.DataFrame): 經過 trackpy 連結且包含角度計算的 DataFrame。
    output_path (str): 輸出的影片路徑 (例如: 'output.mp4')。
    fps (float): 原始影片的幀率 (用於計算前 10 秒對應的幀數)。
    tail_seconds (int): 要繪製過去幾秒內的軌跡拖尾。
    """
    print(f"\n正在渲染追蹤影片... 輸出至: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟原始影片 {video_path}")
        return

    # 取得原始影片的屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 若使用者沒有指定 fps，嘗試從影片讀取
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        fps = video_fps
    
    tail_frames = int(fps * tail_seconds) # 計算要畫前幾幀的軌跡
    
    # 設定影片寫入器 (使用 MP4V 編碼，支援性好)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 建立背景減除器 (為了在畫面上重新找到當下的輪廓)
    # 這裡必須與 extract_positions 使用相同的參數
    fgbg = cv2.createBackgroundSubtractorMOG2(MOG_History, MOG_VarTHR, detectShadows=False)

    # 為了讓每隻眼蟲有固定的顏色，建立一個隨機顏色表
    # 保證相同 particle ID 的顏色在整部影片中都一樣
    unique_particles = df['particle'].unique()
    np.random.seed(42) # 固定隨機種子確保每次跑顏色一樣
    colors = {p: tuple(np.random.randint(50, 255, 3).tolist()) for p in unique_particles}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 只處理 DataFrame 中有記錄的影格
        # 如果使用者之前設定 max_frames=500，這裡也只會畫 500 幀
        if frame_idx > df['frame'].max():
            break

        # --- 1. 重新提取當前影格的輪廓 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fgmask = fgbg.apply(blurred)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 篩選輪廓 (條件需與 extract_positions 一致)
        valid_contours = [cnt for cnt in contours if 15 < cv2.contourArea(cnt) < 400]
        
        # --- 2. 獲取當前影格的所有「有效軌跡資料」 ---
        current_data = df[df['frame'] == frame_idx]
        
        for _, row in current_data.iterrows():
            p_id = int(row['particle'])
            cx, cy = int(row['x']), int(row['y'])
            color = colors.get(p_id, (0, 255, 0)) # 若找不到給預設綠色

            # -- 畫輪廓線 (在 valid_contours 中尋找距離最近的輪廓) --
            min_dist = float('inf')
            best_cnt = None
            for cnt in valid_contours:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cnt_x = int(M['m10'] / M['m00'])
                    cnt_y = int(M['m01'] / M['m00'])
                    # 計算距離
                    dist = (cx - cnt_x)**2 + (cy - cnt_y)**2
                    if dist < min_dist and dist < 400: # 距離要夠近 (20 pixel 內)
                        min_dist = dist
                        best_cnt = cnt
            
            if best_cnt is not None:
                # 畫出該眼蟲的輪廓線 (粗細為 1)
                cv2.drawContours(frame, [best_cnt], -1, color, 1)

            # -- 畫歷史軌跡拖尾 (Tail) --
            # 撈出這隻眼蟲在過去 tail_frames 內的資料
            past_data = df[(df['particle'] == p_id) & 
                           (df['frame'] <= frame_idx) & 
                           (df['frame'] >= frame_idx - tail_frames)]
            
            # 將軌跡座標轉為整數陣列供 cv2.polylines 畫線
            if len(past_data) > 1:
                pts = past_data[['x', 'y']].values.astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                # 畫軌跡線 (粗細為 2)
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
            
            # 標示 ID
            cv2.putText(frame, str(p_id), (cx + 5, cy - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 寫入文字資訊
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Live Tracks: {len(current_data)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        # 寫入影片
        out.write(frame)
        
        if frame_idx % 100 == 0: 
            print(f"已渲染 {frame_idx} 幀...")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ 渲染完成！影片已儲存至: {output_path}")

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
