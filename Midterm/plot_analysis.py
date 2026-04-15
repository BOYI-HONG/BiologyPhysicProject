import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定檔案路徑
# ==========================================
video_file = '/mnt/j/BiologicalPhysics/V1/Euglena_grad_lateral/Euglena_grad_lateral/2023_06_15_Euglena_13/video.h264'
csv_file = 'euglena_500_frames.csv'

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
# 執行主程式
# ==========================================
if __name__ == "__main__":
    generate_background_image(video_file)
    plot_kinematics(csv_file)