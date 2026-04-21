import cv2
import pandas as pd
import numpy as np
import trackpy as tp
import os

# ==========================================
# 策略 B 專用追蹤參數設定 (Global Configs)
# ==========================================
DIAMETER = 15      # 預期眼蟲的直徑 (需為奇數，如 11, 15, 17)
MINMASS = 200      # 斑點的最小總亮度 (原 500 可能太高，調低至 200 以抓取較暗眼蟲)
SEARCH_RANGE = 30  # Trackpy 尋找下一幀同一個體的最大像素距離
MEMORY = 5         # Trackpy 容許眼蟲短暫消失的幀數
THRESHOLD = 10     # 最短有效軌跡長度 (存活超過幾幀才算數)

# ==========================================
# 1. 影像預處理與斑點偵測 (策略 B: tp.locate)
# ==========================================
def extract_positions(video_path, max_frames=-1):
    if max_frames != -1:
        print(f"正在使用 Trackpy 斑點偵測讀取影片 (僅處理前 {max_frames} 影格):\n{video_path}")
    else:
        print(f"正在使用 Trackpy 斑點偵測讀取影片 (處理全影格):\n{video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案，請檢查路徑。")
        return pd.DataFrame()

    data_frames = []
    frame_idx = 0
    
    # --- 計算全域靜態背景 ---
    print("正在計算全域靜態背景...")
    bg_accumulator = None
    
    # 若 max_frames 未指定，則嘗試獲取影片總幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_frames_count = 50
    if max_frames != -1 and max_frames < 50:
        bg_frames_count = max_frames
    elif total_frames > 0 and total_frames < 50:
        bg_frames_count = total_frames

    for _ in range(bg_frames_count):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bg_accumulator is None:
            bg_accumulator = np.zeros_like(gray, dtype=np.float32)
        bg_accumulator += gray
    
    if bg_accumulator is not None:
        global_bg = (bg_accumulator / bg_frames_count).astype(np.uint8)
    else:
        global_bg = None
        
    # 重置影片讀取頭到開頭
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("開始進行斑點偵測 (Blob Detection)...")
    while cap.isOpened():
        if max_frames != -1 and frame_idx >= max_frames: 
            break
            
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 扣除全域靜態背景
        if global_bg is not None:
            diff = cv2.subtract(gray, global_bg)
            diff = cv2.convertScaleAbs(diff, alpha=1.5, beta=0)
        else:
            diff = gray

        # 使用 Trackpy 尋找斑點
        f = tp.locate(diff, diameter=DIAMETER, minmass=MINMASS, invert=False)
        
        if not f.empty:
            f['frame'] = frame_idx
            f['body_angle'] = np.nan 
            f['area'] = f['mass'] 
            data_frames.append(f)
            
        # if frame_idx % 100 == 0: 
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
    t = tp.link_df(df, search_range=SEARCH_RANGE, memory=MEMORY)
    t_filtered = tp.filter_stubs(t, threshold=THRESHOLD)
    
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

# ==========================================
# 4. 驗證影片渲染 (策略 B 專用版)
# ==========================================
def render_tracking_video(video_path, df, output_path, fps=30.0, tail_seconds=10):
    """
    策略 B 專用的渲染函數：在影片上畫出質心(實心圓點)與拖尾。
    """
    print(f"\n正在渲染追蹤影片... 輸出至: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟原始影片 {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        fps = video_fps
    
    tail_frames = int(fps * tail_seconds) 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    unique_particles = df['particle'].unique()
    np.random.seed(42)
    colors = {p: tuple(np.random.randint(50, 255, 3).tolist()) for p in unique_particles}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx > df['frame'].max():
            break

        current_data = df[df['frame'] == frame_idx]
        
        for _, row in current_data.iterrows():
            p_id = int(row['particle'])
            cx, cy = int(row['x']), int(row['y'])
            color = colors.get(p_id, (0, 255, 0))

            # 畫質心 (實心圓點)
            cv2.circle(frame, (cx, cy), radius=3, color=color, thickness=-1)

            # 畫歷史軌跡拖尾
            past_data = df[(df['particle'] == p_id) & 
                           (df['frame'] <= frame_idx) & 
                           (df['frame'] >= frame_idx - tail_frames)]
            
            if len(past_data) > 1:
                pts = past_data[['x', 'y']].values.astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
            
            # 標示 ID
            cv2.putText(frame, str(p_id), (cx + 5, cy - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Live Tracks: {len(current_data)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        out.write(frame)
        
        if frame_idx % 100 == 0: 
            print(f"已渲染 {frame_idx} 幀...")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ 渲染完成！影片已儲存至: {output_path}")