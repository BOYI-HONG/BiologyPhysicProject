import cv2
import pandas as pd
import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt
video_file = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_36/2023_06_26_Euglena_36.h264'
csv_file = 'Euglena_circle_light_2023_06_26_Euglena_36.csv'

# ==========================================
# 1. 影像預處理與座標提取 (限制前 500 幀)
# ==========================================
def extract_positions(video_path):
    print(f"正在讀取影片 (僅處理前 500 影格):\n{video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("錯誤：無法開啟影片檔案，請檢查路徑。")
        return pd.DataFrame()

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
    data = []
    frame_idx = 0
    
    while cap.isOpened():
        # if frame_idx >= 500: 
            # print("已達到 500 影格限制，停止提取。")
            # break
            
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

# ==========================================
# 2. 軌跡連結 (Trackpy) - 徹底解決變數衝突
# ==========================================
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

# ==========================================
# 3. 相位角分析
# ==========================================
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

# ==========================================
# 執行
# ==========================================
if __name__ == "__main__":

    raw_points = extract_positions(video_file)
    
    if not raw_points.empty:
        tracks = link_data(raw_points)
        final_data = calculate_movement_angles(tracks)
        
        # 存檔# 存檔並限制小數點後三位
        final_data.to_csv(csv_file, index=False, float_format='%.3f')
        print("前 %d 幀分析完成！數據存至 euglena_500_frames.csv")
        
        # 視覺化 (使用 headless 模式存成圖片，避免 WSL 顯示問題)
        print("正在繪製軌跡圖並存為 png...")
        plt.figure(figsize=(10, 8))
        # 1. 統計每隻眼蟲出現的次數 (即軌跡長度)
        # 2. 取得前 20 名最長壽的 particle ID
        top_20_particles = final_data['particle'].value_counts().head(20).index

        plt.figure(figsize=(12, 10))
        for p_id in top_20_particles:
            # 提取該 ID 的數據並確保按時間排序
            p_data = final_data[final_data['particle'] == p_id].sort_values('frame')
            plt.plot(p_data['x'], p_data['y'], label=f'ID:{p_id}', alpha=0.8)

        plt.title("Top 20 Longest Trajectories of Euglena")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig('trajectories_top20.png', dpi=300)
        # plt.show()
        print("圖片已存至 trajectories_500.png")
    else:
        print("未抓取到資料。")