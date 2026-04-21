import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns  # <--- 新增這行，用於繪製邊際分佈密度圖

# ==========================================
# 輔助函數：支援吃入 CSV 路徑或直接吃 DataFrame
# ==========================================
def _load_data(data):
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {data}")
            return None
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        print("錯誤：傳入的資料格式不正確，必須是 CSV 路徑或 Pandas DataFrame。")
        return None

# ==========================================
# 1. 生成並儲存背景噪音圖 (Background Model)
# ==========================================
def generate_background_image(video_path, out_dir, num_frames=-1):
    print("正在提取背景模型 (Background Noise Map)...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法讀取影片 {video_path}")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=40, detectShadows=False)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fgbg.apply(blurred)
        frame_idx += 1
        
    bg_img = fgbg.getBackgroundImage()
    cap.release()
    
    if bg_img is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(bg_img, cmap='gray')
        plt.title("Background Illumination & Noise Map")
        plt.colorbar(label="Pixel Intensity (0-255)")
        
        # 存檔至指定的 out_dir
        save_path = os.path.join(out_dir, '01_background_noise_map.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"背景圖已儲存為 {save_path}")
    else:
        print("無法生成背景圖。")

# ==========================================
# 2. 繪製 X, Y, Theta 對應時間 (Frame) 的變化
# ==========================================
def plot_kinematics(data, out_dir):
    print("正在繪製動力學圖表 (Kinematics)...")
    df = _load_data(data)
    if df is None or df.empty: return

    track_lengths = df['particle'].value_counts()
    top_particles = track_lengths.head(5).index.tolist()
    
    if not top_particles:
        print("沒有足夠的軌跡可以繪圖。")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for p_id in top_particles:
        p_data = df[df['particle'] == p_id].sort_values('frame')
        frames = p_data['frame']
        
        axes[0].plot(frames, p_data['x'], linewidth=2, label=f'ID: {p_id}')
        axes[1].plot(frames, p_data['y'], linewidth=2)
        
        theta_degrees = np.degrees(p_data['move_angle'])
        axes[2].plot(frames, theta_degrees, '.', markersize=4, alpha=0.7)

    axes[0].set_title("X Position vs Time (Top 5)")
    axes[0].set_ylabel("X (pixels)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("Y Position vs Time")
    axes[1].set_ylabel("Y (pixels)")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].set_title("Phase Angle (Theta) vs Time")
    axes[2].set_ylabel("Angle (Degrees)")
    axes[2].set_xlabel("Time (Frames)")
    axes[2].set_yticks(np.arange(-180, 181, 90)) 
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(out_dir, '02_kinematics_X_Y_Theta_vs_T.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"動力學圖表已儲存為 {save_path}")

# ==========================================
# 3. 繪製 X, Y, Theta 隨時間的群體統計分布 (2D Histogram)
# ==========================================
def plot_population_statistics(data, out_dir):
    print("正在計算群體密度熱力圖 (2D Histograms)...")
    df = _load_data(data)
    if df is None or df.empty: return

    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    if df_clean.empty:
        print("資料清理後為空，無法繪製熱力圖。")
        return

    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])
    
    max_frame = df_clean['frame'].max()
    x_min, x_max = df_clean['x'].min(), df_clean['x'].max()
    y_min, y_max = df_clean['y'].min(), df_clean['y'].max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    time_bins = min(50, int(max_frame/10)) if max_frame > 0 else 50

    h_x = axes[0].hist2d(df_clean['frame'], df_clean['x'], bins=[time_bins, 40], range=[[0, max_frame], [x_min, x_max]], cmap='inferno')
    axes[0].set_title('Spatial Density Map (X) over Time')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].grid(False)
    fig.colorbar(h_x[3], ax=axes[0]).set_label('Agent Count')

    h_y = axes[1].hist2d(df_clean['frame'], df_clean['y'], bins=[time_bins, 40], range=[[0, max_frame], [y_min, y_max]], cmap='inferno')
    axes[1].set_title('Spatial Density Map (Y) over Time')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].grid(False)
    fig.colorbar(h_y[3], ax=axes[1]).set_label('Agent Count')

    h_t = axes[2].hist2d(df_clean['frame'], df_clean['theta_deg'], bins=[time_bins, 36], range=[[0, max_frame], [-180, 180]], cmap='inferno')
    axes[2].set_title('Phase Angle (\u03b8) Density Heatmap over Time')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].set_xlabel('Time (Frames)')
    axes[2].set_yticks([-180, -90, 0, 90, 180])
    axes[2].grid(False)
    fig.colorbar(h_t[3], ax=axes[2]).set_label('Agent Count')

    plt.tight_layout()
    save_path = os.path.join(out_dir, '03_population_2D_histograms.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"群體 2D 熱力圖已儲存為 {save_path}")

# ==========================================
# 4. 繪製群體散佈圖 (Scatter Plots)
# ==========================================
def plot_population_scatter(data, out_dir):
    print("正在繪製群體散佈圖 (Scatter Plots)...")
    df = _load_data(data)
    if df is None or df.empty: return

    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    if df_clean.empty: return
    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc1 = axes[0].scatter(df_clean['x'], df_clean['y'], c=df_clean['frame'], cmap='viridis', s=2, alpha=0.5)
    axes[0].set_title('Spatial Trajectories (X vs Y)')
    axes[0].set_xlabel('X Position (pixels)')
    axes[0].set_ylabel('Y Position (pixels)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].set_aspect('equal', adjustable='datalim') 
    fig.colorbar(sc1, ax=axes[0]).set_label('Time (Frames)')

    sc2 = axes[1].scatter(df_clean['frame'], df_clean['theta_deg'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.6)
    axes[1].set_title('Population Phase Angle (\u03b8) vs Time')
    axes[1].set_xlabel('Time (Frames)')
    axes[1].set_ylabel('Angle (Degrees)')
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(out_dir, '04_population_scatter_plots.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"群體散佈圖已儲存為 {save_path}")

# ==========================================
# 5. 繪製原始數據散佈圖 (Raw Scatter Plots)
# ==========================================
def plot_raw_scatter(data, out_dir):
    print("正在繪製原始數據散佈圖 (Raw Scatter Plots)...")
    df = _load_data(data)
    if df is None or df.empty: return

    df_clean = df.dropna(subset=['frame', 'x', 'y', 'move_angle']).copy()
    if df_clean.empty: return
    df_clean['theta_deg'] = np.degrees(df_clean['move_angle'])

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    axes[0].scatter(df_clean['frame'], df_clean['x'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[0].set_title('Raw Scatter: X Position over Time')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].scatter(df_clean['frame'], df_clean['y'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[1].set_title('Raw Scatter: Y Position over Time')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    axes[2].scatter(df_clean['frame'], df_clean['theta_deg'], c=df_clean['particle'], cmap='tab20', s=2, alpha=0.5)
    axes[2].set_title('Raw Scatter: Phase Angle (\u03b8) over Time')
    axes[2].set_ylabel('Angle (Degrees)')
    axes[2].set_xlabel('Time (Frames)')
    axes[2].set_yticks([-180, -90, 0, 90, 180])
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path_3x1 = os.path.join(out_dir, '05_population_raw_scatter_3x1.png')
    plt.savefig(save_path_3x1, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df_clean['x'], df_clean['y'], c=df_clean['frame'], cmap='viridis', s=2, alpha=0.5)
    plt.title('Real Spatial Trajectories (X vs Y)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.axis('equal') 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.colorbar(sc).set_label('Time (Frames)')
    
    save_path_xy = os.path.join(out_dir, '06_population_raw_scatter_XY.png')
    plt.savefig(save_path_xy, dpi=300)
    plt.close()
    
    print(f"原始散佈圖已儲存為 {save_path_3x1} 與 {save_path_xy}")


# ==========================================
# 6. 繪製速度與角速度隨時間的變化 (Top 5 個體)
# ==========================================
def plot_speed_kinematics(data, out_dir):
    print("正在繪製個體速度與角速度圖表...")
    df = _load_data(data)
    if df is None or df.empty: return

    # 計算物理量
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angular_speed'] = np.degrees(np.abs(df['d_phi']))

    track_lengths = df['particle'].value_counts()
    top_particles = track_lengths.head(5).index.tolist()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for p_id in top_particles:
        p_data = df[df['particle'] == p_id].sort_values('frame')
        frames = p_data['frame']
        
        # 畫線速度
        axes[0].plot(frames, p_data['speed'], linewidth=1.5, label=f'ID: {p_id}', alpha=0.8)
        # 畫角速度
        axes[1].plot(frames, p_data['angular_speed'], linewidth=1.5, alpha=0.8)

    axes[0].set_title("Linear Speed |v| vs Time (Top 5)")
    axes[0].set_ylabel("Speed (pixels/frame)")
    axes[0].legend(loc='upper right', ncol=2, fontsize='small')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("Angular Speed |\u03c9| vs Time (Top 5)")
    axes[1].set_ylabel("Angular Speed (deg/frame)")
    axes[1].set_xlabel("Time (Frames)")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(out_dir, '07_speed_kinematics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"速度動力學圖表已儲存為 {save_path}")

# ==========================================
# 7. 繪製速度與角速度的群體統計分布 (2D Histogram)
# ==========================================
def plot_speed_distributions(data, out_dir):
    print("正在計算速度與角速度群體熱力圖...")
    df = _load_data(data)
    if df is None or df.empty: return

    # 清洗與計算
    df_clean = df.dropna(subset=['dx', 'dy', 'd_phi']).copy()
    df_clean['speed'] = np.sqrt(df_clean['dx']**2 + df_clean['dy']**2)
    df_clean['angular_speed'] = np.degrees(np.abs(df_clean['d_phi']))
    
    max_frame = df_clean['frame'].max()
    time_bins = min(50, int(max_frame/10)) if max_frame > 0 else 50

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 子圖 1: 線速度熱力圖 ---
    h_v = axes[0].hist2d(df_clean['frame'], df_clean['speed'], 
                         bins=[time_bins, 30], cmap='magma')
    axes[0].set_title('Population Linear Speed Distribution over Time')
    axes[0].set_ylabel('Speed (pixels/frame)')
    fig.colorbar(h_v[3], ax=axes[0]).set_label('Agent Count')

    # --- 子圖 2: 角速度熱力圖 ---
    h_w = axes[1].hist2d(df_clean['frame'], df_clean['angular_speed'], 
                         bins=[time_bins, 30], cmap='magma')
    axes[1].set_title('Population Angular Speed Distribution over Time')
    axes[1].set_ylabel('Angular Speed (deg/frame)')
    axes[1].set_xlabel('Time (Frames)')
    fig.colorbar(h_w[3], ax=axes[1]).set_label('Agent Count')

    plt.tight_layout()
    save_path = os.path.join(out_dir, '08_speed_population_histograms.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"速度群體統計圖表已儲存為 {save_path}")

# ==========================================
# 8. 繪製群體平均時間序列與標準差陰影帶
# ==========================================
def plot_population_time_series_with_shade(data, out_dir, light_span=None):
    """
    繪製群體平均速度與角速度隨時間變化的曲線，包含標準差陰影帶。
    light_span: tuple, 例如 (60, 120)，代表要在哪兩個影格之間畫出紅色背景(光刺激)。
    """
    print("正在繪製群體時間序列與標準差陰影帶...")
    df = _load_data(data)
    if df is None or df.empty: return

    # 計算物理量
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angular_speed'] = np.abs(df['d_phi']) # 論文圖通常用弧度或度，這裡先維持弧度，如果太小可以轉角度

    # 依照影格 (frame) 進行群組，計算平均值與標準差
    stats = df.groupby('frame').agg({
        'speed': ['mean', 'std'], 
        'angular_speed': ['mean', 'std']
    }).dropna()

    frames = stats.index

    # 建立 2x1 的子圖，版面比例與論文圖相似
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- 上圖：線速度 v ---
    v_mean = stats['speed']['mean']
    v_std = stats['speed']['std']
    
    axes[0].plot(frames, v_mean, color='blue', linewidth=2.5) # 深藍色實線
    # 畫出淺藍色陰影 (標準差)，將下限限制在 0 以符合物理意義
    axes[0].fill_between(frames, np.clip(v_mean - v_std, 0, None), v_mean + v_std, 
                         color='blue', alpha=0.3, linewidth=0)
    
    axes[0].set_ylabel(r'$v$ [pixels/frame]', fontsize=16) # 使用 LaTeX 數學字體
    axes[0].tick_params(labelsize=12)

    # --- 下圖：角速度 |ω| ---
    w_mean = stats['angular_speed']['mean']
    w_std = stats['angular_speed']['std']
    
    axes[1].plot(frames, w_mean, color='blue', linewidth=2.5)
    axes[1].fill_between(frames, np.clip(w_mean - w_std, 0, None), w_mean + w_std, 
                         color='blue', alpha=0.3, linewidth=0)
    
    axes[1].set_ylabel(r'$|\omega|$ [rad/frame]', fontsize=16)
    axes[1].set_xlabel(r'$t$ [frames]', fontsize=16)
    axes[1].tick_params(labelsize=12)

    # --- 加上光刺激的紅色背景區間 ---
    if light_span:
        start_frame, end_frame = light_span
        for ax in axes:
            # 畫出紅色半透明背景
            ax.axvspan(start_frame, end_frame, color='red', alpha=0.25, lw=0)

    # 美化邊框與版面
    for ax in axes:
        ax.set_xlim(frames.min(), frames.max())
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '09_population_time_series_shaded.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 群體時間序列帶狀圖已儲存為 {save_path}")


# ==========================================
# 9. 繪製平均速度與角速度之二維散佈圖 (附邊際密度分佈)
# ==========================================
def plot_mean_speed_scatter_with_marginals(data, out_dir):
    """
    計算每隻眼蟲的「平均線速度」與「平均角速度」，並畫出帶有 KDE 密度曲線的聯合散佈圖。
    """
    print("正在繪製聯合散佈圖與邊際密度分佈 (JointPlot)...")
    df = _load_data(data)
    if df is None or df.empty: return

    # 計算物理量
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['angular_speed'] = np.abs(df['d_phi'])

    # 依照 particle ID 進行群組，計算每隻個體「整段生命週期」的平均值
    particle_stats = df.groupby('particle').agg({
        'speed': 'mean', 
        'angular_speed': 'mean'
    }).dropna()

    # 使用 seaborn 的 JointGrid 來建立主圖與邊緣圖的架構
    g = sns.JointGrid(data=particle_stats, x='speed', y='angular_speed', height=8, ratio=5)

    # 1. 畫主圖的散佈圖 (空心藍色圓圈)
    g.plot_joint(plt.scatter, facecolors='none', edgecolors='blue', s=40, alpha=0.7)

    # 2. 畫邊際分佈的核密度估計曲線 (KDE)
    g.plot_marginals(sns.kdeplot, color='blue', linewidth=3)

    # 3. 設定標籤 (使用 LaTeX 格式匹配你的範例圖)
    g.ax_joint.set_xlabel(r'mean $v$ [pixels/frame]', fontsize=16)
    g.ax_joint.set_ylabel(r'mean $|\omega|$ [rad/frame]', fontsize=16)
    g.ax_joint.tick_params(labelsize=12)

    # 自動調整範圍，讓它從 0 開始
    g.ax_joint.set_xlim(left=0)
    g.ax_joint.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(out_dir, '10_scatter_meanOnTime.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 帶邊際分佈的二維散佈圖已儲存為 {save_path}")

# ==========================================
# 測試區塊 (僅供單獨執行此檔案時測試用)
# ==========================================
if __name__ == "__main__":
    print("此為繪圖模組，請透過主程式 (main.py) 呼叫。")