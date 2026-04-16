import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定檔案路徑
# ==========================================
video_file = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_36/2023_06_26_Euglena_36.h264'
csv_file = 'Euglena_circle_light_2023_06_26_Euglena_36.csv'

# ==========================================
# 1. 生成並儲存背景噪音圖 (Background Model)
# ==========================================
def generate_background_image(video_path, num_frames=500):
    print("正在提取背景模型 (Background Noise Map)...")
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=40, detectShadows=False)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fgbg.apply(blurred)
        frame_idx += 1
        
    # 獲取最終的背景模型 (這張圖會顯示不變的背景光場與載玻片髒污)
    bg_img = fgbg.getBackgroundImage()
    cap.release()
    
    if bg_img is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(bg_img, cmap='gray')
        plt.title("Background Illumination & Noise Map")
        plt.colorbar(label="Pixel Intensity (0-255)")
        plt.savefig('background_noise_map.png', dpi=300)
        plt.close()
        print("背景圖已儲存為 background_noise_map.png")
    else:
        print("無法生成背景圖。")

# ==========================================
# 2. 繪製 X, Y, Theta 對應時間 (Frame) 的變化
# ==========================================
def plot_kinematics(csv_path):
    print("正在讀取 CSV 進行動力學繪圖...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 {csv_path}，請確認追蹤程式已成功跑完。")
        return

    # 找出軌跡最長 (存活幀數最多) 的前 5 隻眼蟲
    track_lengths = df['particle'].value_counts()
    top_particles = track_lengths.head(5).index.tolist()
    
    print(f"挑選出軌跡最長的 5 隻眼蟲進行詳細分析，ID: {top_particles}")

    # 建立 3x1 的子圖表 (共享 X 軸)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for p_id in top_particles:
        p_data = df[df['particle'] == p_id].sort_values('frame')
        frames = p_data['frame']
        
        # 畫 X vs T
        axes[0].plot(frames, p_data['x'], linewidth=2, label=f'ID: {p_id}')
        # 畫 Y vs T
        axes[1].plot(frames, p_data['y'], linewidth=2)
        
        # 畫 Theta vs T (將弧度轉為角度方便人類閱讀)
        # 由於角度在 -180 到 180 之間跳動，我們用散點或細線畫會比較清楚
        theta_degrees = np.degrees(p_data['move_angle'])
        axes[2].plot(frames, theta_degrees, '.', markersize=4, alpha=0.7)

    # 設定圖表格式
    axes[0].set_title("X Position vs Time")
    axes[0].set_ylabel("X (pixels)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("Y Position vs Time")
    axes[1].set_ylabel("Y (pixels)")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].set_title("Phase Angle (Theta) vs Time")
    axes[2].set_ylabel("Angle (Degrees)")
    axes[2].set_xlabel("Time (Frames)")
    # 設定 Y 軸範圍為標準角度
    axes[2].set_yticks(np.arange(-180, 181, 90)) 
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('kinematics_X_Y_Theta_vs_T.png', dpi=300)
    plt.close()
    print("動力學圖表已儲存為 kinematics_X_Y_Theta_vs_T.png")

# ==========================================
# 3. 繪製 X, Y, Theta 隨時間的群體統計分布
# ==========================================
# ==========================================
# 3. 繪製 X, Y, Theta 隨時間的群體統計分布 (2D Histogram)
# ==========================================
def plot_population_statistics(csv_path):
    print("正在計算群體密度熱力圖 (2D Histograms)...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 {csv_path}。")
        return

    # 清除包含 NaN 的資料列，確保畫圖不報錯
    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    
    # 轉換角度
    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])
    
    max_frame = df_clean['frame'].max()
    
    # 取得空間的最大與最小邊界，留一點 padding
    x_min, x_max = df_clean['x'].min(), df_clean['x'].max()
    y_min, y_max = df_clean['y'].min(), df_clean['y'].max()

    # 建立 3x1 子圖 (稍微加高一點以容納三個 Colorbar)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 共用的時間軸切分數量
    time_bins = min(50, int(max_frame/10)) if max_frame > 0 else 50

    # --- 子圖 1: X 的群體密度分佈 ---
    h_x = axes[0].hist2d(df_clean['frame'], df_clean['x'], 
                         bins=[time_bins, 40], 
                         range=[[0, max_frame], [x_min, x_max]], 
                         cmap='inferno')
    axes[0].set_title('Spatial Density Map (X) over Time')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].grid(False) # 熱力圖通常不需要網格
    # 加上顏色條 (Colorbar)
    cbar_x = fig.colorbar(h_x[3], ax=axes[0])
    cbar_x.set_label('Agent Count')

    # --- 子圖 2: Y 的群體密度分佈 ---
    h_y = axes[1].hist2d(df_clean['frame'], df_clean['y'], 
                         bins=[time_bins, 40], 
                         range=[[0, max_frame], [y_min, y_max]], 
                         cmap='inferno')
    axes[1].set_title('Spatial Density Map (Y) over Time')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].grid(False)
    cbar_y = fig.colorbar(h_y[3], ax=axes[1])
    cbar_y.set_label('Agent Count')

    # --- 子圖 3: Theta 的群體密度分佈 ---
    # 角度切 36 等分 (代表每 10 度一個 bin)
    h_t = axes[2].hist2d(df_clean['frame'], df_clean['theta_deg'], 
                         bins=[time_bins, 36], 
                         range=[[0, max_frame], [-180, 180]], 
                         cmap='inferno')
    axes[2].set_title('Phase Angle (\u03b8) Density Heatmap over Time')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].set_xlabel('Time (Frames)')
    axes[2].set_yticks([-180, -90, 0, 90, 180])
    axes[2].grid(False)
    cbar_t = fig.colorbar(h_t[3], ax=axes[2])
    cbar_t.set_label('Agent Count')

    plt.tight_layout()
    plt.savefig('population_2D_histograms.png', dpi=300)
    plt.close()
    print("群體 2D 熱力圖已儲存為 population_2D_histograms.png")

# ==========================================
# 4. 繪製群體散佈圖 (Scatter Plots)
# ==========================================
def plot_population_scatter(csv_path):
    print("正在繪製群體散佈圖 (Scatter Plots)...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 {csv_path}。")
        return

    # 清除空值
    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 圖 1: 空間軌跡散佈圖 (X vs Y)，顏色代表時間 (Frame) ---
    # 使用散佈圖畫出所有點，顏色越亮/越暖代表時間越往後
    sc1 = axes[0].scatter(df_clean['x'], df_clean['y'], 
                          c=df_clean['frame'], cmap='viridis', 
                          s=2, alpha=0.5) # s是點的大小, alpha是透明度
    
    axes[0].set_title('Spatial Trajectories (X vs Y)')
    axes[0].set_xlabel('X Position (pixels)')
    axes[0].set_ylabel('Y Position (pixels)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 保持 X 軸和 Y 軸的比例一致，這樣看到的軌跡才不會變形
    axes[0].set_aspect('equal', adjustable='datalim') 
    
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Time (Frames)')

    # --- 圖 2: 群體相位角散佈圖 (Frame vs Theta) ---
    # 相比於熱力圖，散佈圖更能看出單一軌跡在角度上的「連續跳動」
    sc2 = axes[1].scatter(df_clean['frame'], df_clean['theta_deg'], 
                          c=df_clean['particle'], cmap='tab20', # 用不同顏色區分不同隻眼蟲
                          s=2, alpha=0.6)
    
    axes[1].set_title('Population Phase Angle (\u03b8) vs Time')
    axes[1].set_xlabel('Time (Frames)')
    axes[1].set_ylabel('Angle (Degrees)')
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('population_scatter_plots.png', dpi=300)
    plt.close()
    print("群體散佈圖已儲存為 population_scatter_plots.png")

# ==========================================
# 4. 繪製原始數據散佈圖 (Raw Scatter Plots)
# ==========================================
def plot_raw_scatter(csv_path):
    print("正在繪製原始數據散佈圖 (Raw Scatter Plots)...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 {csv_path}。")
        return

    # 清除包含 NaN 的列
    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])

    # ---------------------------------------------------------
    # 第一張圖：完全對照 2D Hist 的 3x1 時間序列散佈圖
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 為了區分不同的眼蟲，我們用 particle ID 來上色 (使用 tab20 離散色票)
    # s=2 代表點的大小，alpha=0.5 代表半透明，這樣點重疊時才不會變成死黑一塊
    
    # 子圖 1: X vs Time
    axes[0].scatter(df_clean['frame'], df_clean['x'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[0].set_title('Raw Scatter: X Position over Time')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 子圖 2: Y vs Time
    axes[1].scatter(df_clean['frame'], df_clean['y'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[1].set_title('Raw Scatter: Y Position over Time')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 子圖 3: Theta vs Time
    axes[2].scatter(df_clean['frame'], df_clean['theta_deg'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[2].set_title('Raw Scatter: Phase Angle (\u03b8) over Time')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].set_xlabel('Time (Frames)')
    axes[2].set_yticks([-180, -90, 0, 90, 180])
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('population_raw_scatter_3x1.png', dpi=300)
    plt.close()
    print("3x1 時間序列散佈圖已儲存為 population_raw_scatter_3x1.png")

    # ---------------------------------------------------------
    # 第二張圖：空間真實軌跡散佈圖 (X vs Y)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # 這裡我們用時間 (frame) 來上色，顏色越暖(黃)代表時間越後面
    sc = plt.scatter(df_clean['x'], df_clean['y'], c=df_clean['frame'], cmap='viridis', s=2, alpha=0.5)
    
    plt.title('Real Spatial Trajectories (X vs Y)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    
    # 【關鍵設定】：強制設定 X 軸與 Y 軸的比例為 1:1
    # 這樣畫出來的圖才不會變形，完美還原顯微鏡下的真實幾何形狀
    plt.axis('equal') 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (Frames)')
    
    plt.savefig('population_raw_scatter_XY.png', dpi=300)
    plt.close()
    print("空間軌跡散佈圖已儲存為 population_raw_scatter_XY.png")

# ==========================================
# 執行主程式
# ==========================================
if __name__ == "__main__":
    # 原本的
    generate_background_image(video_file)
    plot_kinematics(csv_file)
    
    # 新增的群體統計
    plot_population_statistics(csv_file)
    plot_population_scatter(csv_file)
    plot_raw_scatter(csv_file)