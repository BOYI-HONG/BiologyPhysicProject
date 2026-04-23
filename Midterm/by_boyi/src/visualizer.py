import cv2
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

def save_density_image(density_matrix, output_filename):
    """(保留你原本的密度圖功能) 將密度矩陣正規化並儲存為圖片"""
    print(f"🎨 [Visualizer] 正在生成視覺化圖檔: {output_filename}")
    normalized = cv2.normalize(density_matrix, None, 0, 255, cv2.NORM_MINMAX)
    img_8bit = normalized.astype(np.uint8)
    cv2.imwrite(str(output_filename), img_8bit)

# =========================================================
# 巨觀視覺化：時間序列拼圖 (Time-lapse Montage)
# =========================================================
def create_time_series_canvas(image_paths, output_filename, max_images=25, cols=5):
    """將多張灰階影像按時間序列拼貼成一張大畫布。自動進行均勻抽樣與縮圖。"""
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print("  ❌ [Visualizer] 沒有圖片可以繪製畫布！")
        return

    # 1. 均勻抽樣
    if total_imgs > max_images:
        indices = np.linspace(0, total_imgs - 1, max_images, dtype=int)
        sampled_paths = [image_paths[i] for i in indices]
        print(f"  🎨 [Visualizer] 圖片過多 ({total_imgs}張)，自動均勻抽取 {max_images} 張繪製時間序列...")
    else:
        sampled_paths = image_paths
        print(f"  🎨 [Visualizer] 正在繪製 {total_imgs} 張時間序列影像...")

    # 2. 決定排版與縮圖大小
    THUMBNAIL_WIDTH = 384
    THUMBNAIL_HEIGHT = 216 
    
    rows = math.ceil(len(sampled_paths) / cols)
    canvas_width = cols * THUMBNAIL_WIDTH
    canvas_height = rows * THUMBNAIL_HEIGHT
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 3. 把圖片一張張貼上去
    for idx, img_path in enumerate(sampled_paths):
        row = idx // cols
        col = idx % cols
        
        y_start = row * THUMBNAIL_HEIGHT
        y_end = y_start + THUMBNAIL_HEIGHT
        x_start = col * THUMBNAIL_WIDTH
        x_end = x_start + THUMBNAIL_WIDTH
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img_resized = cv2.resize(img, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
            label = img_path.stem 
            
            cv2.rectangle(img_resized, (5, 5), (140, 35), (0, 0, 0), -1)
            cv2.putText(img_resized, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            canvas[y_start:y_end, x_start:x_end] = img_resized

    # 4. 存檔
    cv2.imwrite(str(output_filename), canvas)
    print(f"  ✅ [Visualizer] 時間序列畫布已儲存為: {output_filename}")


# =========================================================
# 微觀視覺化：繪製動力學圖表 (Kinematics Plot)
# =========================================================
def plot_population_kinematics(time_series_data, output_filename, light_on_range=(60, 120)):
    """
    繪製論文 Fig 2d 同款圖表：速度 v 與絕對角速度 |w| 對時間的關係圖
    """
    print(f"🎨 [Visualizer] 正在生成動力學圖表: {output_filename}")
    
    times = time_series_data['time']
    v_median, v_10, v_90 = [], [], []
    w_median, w_10, w_90 = [], [], []
    
    # 計算每個時間點的統計量 (Median, 10th, 90th percentile)
    for v_list in time_series_data['v']:
        if len(v_list) > 0:
            v_median.append(np.median(v_list))
            v_10.append(np.percentile(v_list, 10))
            v_90.append(np.percentile(v_list, 90))
        else:
            v_median.append(np.nan); v_10.append(np.nan); v_90.append(np.nan)
            
    for w_list in time_series_data['w']:
        if len(w_list) > 0:
            w_median.append(np.median(w_list))
            w_10.append(np.percentile(w_list, 10))
            w_90.append(np.percentile(w_list, 90))
        else:
            w_median.append(np.nan); w_10.append(np.nan); w_90.append(np.nan)

    # 建立畫布與子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # ---- 繪製 Velocity (v) ----
    ax1.plot(times, v_median, color='blue', linewidth=2, label='Median')
    ax1.fill_between(times, v_10, v_90, color='blue', alpha=0.3, label='10th-90th percentile')
    # 判斷是否需要繪製光照區域
    if light_on_range:
        ax1.axvspan(light_on_range[0], light_on_range[1], color='red', alpha=0.4, label='Illumination')
    
    ax1.set_ylabel(r'$v$ ($\mu m / s$)', fontsize=12)
    ax1.set_ylim(0, 120)
    ax1.set_xlim(0, max(times) if len(times) > 0 else 180)
    
    # ---- 繪製 Angular Velocity (|w|) ----
    ax2.plot(times, w_median, color='blue', linewidth=2)
    ax2.fill_between(times, w_10, w_90, color='blue', alpha=0.3)
    if light_on_range:
        ax2.axvspan(light_on_range[0], light_on_range[1], color='red', alpha=0.4)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel(r'$|\omega|$ (rad / s)', fontsize=12)
    # ax2.set_ylim(0, 2.0)
    
    plt.tight_layout()
    plt.savefig(str(output_filename), dpi=300)
    plt.close()
    print(f"  ✅ [Visualizer] 圖表已儲存為: {output_filename}")