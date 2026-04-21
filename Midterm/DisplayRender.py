import cv2
import pandas as pd
import numpy as np
import trackpy as tp
import os
import config  # <--- 匯入集中管理的設定檔

# ==========================================
# 4. 驗證影片渲染 (mp4v + 解析度縮小 50%)
# ==========================================
def render_tracking_video(video_path, df, output_path, fps=30.0, tail_seconds=10):
    """
    讀取原始影片並重新找到輪廓，疊加歷史軌跡與 ID。
    為了節省空間，將輸出影片長寬縮小為 50%。
    """
    print(f"\n正在渲染追蹤影片... 輸出至: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟原始影片 {video_path}")
        return

    # 取得原始長寬
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 計算新的長寬 (縮小 50%)
    new_width = orig_width // 2
    new_height = orig_height // 2
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0: fps = video_fps
    
    tail_frames = int(fps * tail_seconds) 
    
    # 退回最安全的 mp4v 編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # VideoWriter 使用新的縮小尺寸
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    unique_particles = df['particle'].unique()
    np.random.seed(42)
    colors = {p: tuple(np.random.randint(50, 255, 3).tolist()) for p in unique_particles}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx > df['frame'].max(): break

        # --- 影像處理與找輪廓 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [cnt for cnt in contours if config.AREA_MIN < cv2.contourArea(cnt) < config.AREA_MAX]

        current_data = df[df['frame'] == frame_idx]
        
        # --- 在「原始大小」的 frame 上畫圖，座標才不會錯亂 ---
        for _, row in current_data.iterrows():
            p_id = int(row['particle'])
            cx, cy = int(row['x']), int(row['y'])
            color = colors.get(p_id, (0, 255, 0))

            min_dist = float('inf')
            best_cnt = None
            for cnt in valid_contours:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cnt_x = int(M['m10'] / M['m00'])
                    cnt_y = int(M['m01'] / M['m00'])
                    dist = (cx - cnt_x)**2 + (cy - cnt_y)**2
                    if dist < min_dist and dist < 400:
                        min_dist = dist
                        best_cnt = cnt
            
            if best_cnt is not None:
                cv2.drawContours(frame, [best_cnt], -1, color, 1)
            else:
                cv2.circle(frame, (cx, cy), radius=2, color=color, thickness=-1)

            past_data = df[(df['particle'] == p_id) & 
                           (df['frame'] <= frame_idx) & 
                           (df['frame'] >= frame_idx - tail_frames)]
            
            if len(past_data) > 1:
                pts = past_data[['x', 'y']].values.astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
            
            cv2.putText(frame, str(p_id), (cx + 5, cy - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Live Tracks: {len(current_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        # --- 將畫好圖的畫面縮小一半 ---
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 寫入影片
        out.write(resized_frame)
        
        if frame_idx % 100 == 0: 
            print(f"已渲染 {frame_idx} 幀...")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ 渲染完成！影片已縮小並儲存至: {output_path}")