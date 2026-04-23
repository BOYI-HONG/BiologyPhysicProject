import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import csv

# =========================================================
# 1. 前處理：批次影像萃取紅光通道 (I/O 多執行緒加速)
# =========================================================
def batch_convert_to_grayscale(image_paths, output_folder):
    """
    將傳入的所有圖片轉換為單通道，專門萃取「紅色通道」以對應紅光暗視野顯微鏡，
    並極速儲存到指定資料夾。
    """
    if not image_paths:
        print("  ❌ 沒有圖片可以轉換！")
        return []

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _convert_and_save(img_path):
        # 1. 以彩色 (BGR) 模式讀取原始影像
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is not None:
            # 2. 論文 4.4 節：只擷取紅色通道 (OpenCV 順序為 B=0, G=1, R=2)
            red_channel = img[:, :, 2]

            # 3. 組合新的儲存路徑並存檔
            save_path = out_dir / img_path.name
            cv2.imwrite(str(save_path), red_channel)
            return save_path
        return None

    print(f"  🎞️ [Analysis] 啟動多執行緒：正在將 {len(image_paths)} 張影像萃取紅光通道...")

    processed_paths = []
    # 使用 ThreadPoolExecutor 來解決讀寫硬碟的 I/O 瓶頸
    with ThreadPoolExecutor(max_workers=8) as executor:
        for result_path in executor.map(_convert_and_save, image_paths):
            if result_path:
                processed_paths.append(result_path)

    print(f"  ✅ [Analysis] 轉換完畢！已儲存至: {out_dir}")
    return processed_paths

# =========================================================
# 2. 巨觀分析：密度圖 (維持單核，因為要聚合巨大矩陣)
# =========================================================
def calculate_density_map_fast(image_paths, steady_state_count=200):
    """
    計算穩態階段的時間平均密度矩陣 (採用背景預讀取加速)
    """
    total_frames = len(image_paths)
    steady_state_frames = image_paths[-steady_state_count:] if total_frames > steady_state_count else image_paths

    first_img = cv2.imread(str(steady_state_frames[0]), cv2.IMREAD_GRAYSCALE)
    accumulated_matrix = np.zeros_like(first_img, dtype=np.float32)

    def _read_image(path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    print(f"  🧠 [Analysis] 計算 {len(steady_state_frames)} 張穩態影像的時間平均...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for img in executor.map(_read_image, steady_state_frames):
            if img is not None:
                accumulated_matrix += img

    average_matrix = accumulated_matrix / len(steady_state_frames)
    return average_matrix

# =========================================================
# 3. 微觀分析：粒子追蹤與動力學引擎 (混合平行架構 + Top-Hat)
# =========================================================
def extract_trajectories_and_kinematics(image_paths, fps=2.0, pixel_size=1.25):
    """
    執行 Particle Tracking 並計算每個 particle 的 v 與 |w|
    具備多執行緒 Top-Hat 濾波，能克服漸亮與光照不均。
    """
    print(f"  🔬 [Analysis] 啟動多核心粒子追蹤引擎，處理 {len(image_paths)} 張影像...")
    dt = 1.0 / fps

    # --- A. 🚀 加速核心：多執行緒 Top-Hat 萃取質心 ---
    def _extract_centroids_from_frame(img_path):
        # 讀取已經轉好且只有紅光訊號的單通道圖
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return []

        # 1. 形態學 Top-Hat 轉換 (最強的光照不均殺手)
        # 設定 kernel 大小為 21x21 (略大於眼蟲直徑，這樣能完美保留蟲體，消除漸亮背景)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        # 2. 高斯模糊降低感光元件雜訊
        blurred = cv2.GaussianBlur(tophat, (3, 3), 0)

        # 3. 簡單二值化 (將閾值提高到 30，確保只抓到真正的超亮眼蟲)
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 放寬面積限制 (10~2000)，確保蟲體變形時不會漏抓
            if 10 < area < 2000:  
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    centroids.append((cx, cy))
        return centroids

    print("    👁️  [CV] 正在使用 Top-Hat 平行萃取所有影格的物件質心...")
    cores = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=cores) as executor:
        all_frames_centroids = list(executor.map(_extract_centroids_from_frame, image_paths))

    # --- B. 單核心序列化處理：匈牙利演算法軌跡配對 ---
    print("    🔗 [Tracking] 正在進行跨影格軌跡配對...")
    active_tracks = {}
    next_id = 0
    max_distance = 20.0 

    for t_idx, current_centroids in enumerate(all_frames_centroids):
        current_active_ids = [tid for tid, trk in active_tracks.items() if trk['active']]

        if len(current_active_ids) == 0:
            for cx, cy in current_centroids:
                active_tracks[next_id] = {'positions': [(t_idx, cx, cy)], 'active': True}
                next_id += 1
        elif len(current_centroids) > 0:
            prev_centroids = [active_tracks[tid]['positions'][-1][1:3] for tid in current_active_ids]

            cost_matrix = np.zeros((len(prev_centroids), len(current_centroids)))
            for i, p1 in enumerate(prev_centroids):
                for j, p2 in enumerate(current_centroids):
                    cost_matrix[i, j] = np.hypot(p1[0]-p2[0], p1[1]-p2[1])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_current = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_distance:
                    tid = current_active_ids[r]
                    active_tracks[tid]['positions'].append((t_idx, current_centroids[c][0], current_centroids[c][1]))
                    assigned_current.add(c)
                else:
                    active_tracks[current_active_ids[r]]['active'] = False

            for j, (cx, cy) in enumerate(current_centroids):
                if j not in assigned_current:
                    active_tracks[next_id] = {'positions': [(t_idx, cx, cy)], 'active': True}
                    next_id += 1

        for tid in current_active_ids:
            if active_tracks[tid]['positions'][-1][0] < t_idx:
                active_tracks[tid]['active'] = False

    # --- C. 物理動力學計算 ---
    print("    🧮 [Kinematics] 計算速度與角速度...")
    time_series_data = {'time': np.arange(len(image_paths)) * dt, 'v': [], 'w': []}
    for _ in range(len(image_paths)):
        time_series_data['v'].append([])
        time_series_data['w'].append([])

    raw_trajectories = []

    for tid, trk in active_tracks.items():
        pos = trk['positions']
        if len(pos) < 5: continue # 排除雜訊軌跡

        # 1. 整理原始軌跡輸出
        for p in pos:
            t_idx, cx, cy = p
            raw_trajectories.append({
                'particle_id': tid,
                'time_s': t_idx * dt,
                'x_um': cx * pixel_size,
                'y_um': cy * pixel_size
            })

        # 2. 進行運動學數學運算
        t_indices = [p[0] for p in pos]
        pts = np.array([[p[1], p[2]] for p in pos]) * pixel_size

        velocities = np.diff(pts, axis=0) / dt
        kernel = np.ones(3) / 3.0

        if len(velocities) >= 3:
            smooth_vx = np.convolve(velocities[:, 0], kernel, mode='same')
            smooth_vy = np.convolve(velocities[:, 1], kernel, mode='same')
            smooth_v_mag = np.hypot(smooth_vx, smooth_vy)

            w_vals = [0.0] 
            for k in range(1, len(smooth_vx)):
                v_prev = np.array([smooth_vx[k-1], smooth_vy[k-1]])
                v_curr = np.array([smooth_vx[k], smooth_vy[k]])

                dot_p = np.dot(v_prev, v_curr)
                if np.hypot(*v_prev) < 1e-5 or np.hypot(*v_curr) < 1e-5:
                    w_vals.append(0.0)
                else:
                    cross_p = np.cross(v_prev, v_curr)
                    w = (1.0 / dt) * np.arctan2(cross_p, dot_p)
                    w_vals.append(np.abs(w)) 

            for idx, t_idx in enumerate(t_indices[1:]): 
                time_series_data['v'][t_idx].append(smooth_v_mag[idx])
                time_series_data['w'][t_idx].append(w_vals[idx])

    print("  ✅ [Analysis] 動力學數據計算完成！")
    return time_series_data, raw_trajectories

# =========================================================
# 4. 資料輸出模組
# =========================================================
def save_trajectories_to_csv(trajectories, output_file):
    """將粒子的 x, y 時間序列儲存為 CSV 檔案"""
    if not trajectories:
        print("  ⚠️ [Data] 沒有軌跡資料可以儲存。")
        return

    print(f"  💾 [Data] 正在將軌跡資料儲存至: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['particle_id', 'time_s', 'x_um', 'y_um'])
        writer.writeheader()
        writer.writerows(trajectories)