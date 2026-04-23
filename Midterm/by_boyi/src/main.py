import subprocess
from pathlib import Path
from data_loader import DataLoader
import analysis
import visualizer

# ==========================================
# 實驗清單
# ==========================================
ROOT_FOLDERS = [
    "Euglena_BCL",
    "Euglena_OFF",
    "Euglena_half_half",
    "Euglena_ramp",
    "Euglena_switch_1",
    "Euglena_switch_10",
]

def get_dynamic_image_path(root_name, remote_base="gdrive:Biophysics_projects"):
    """動態尋找遠端路徑"""
    search_path = f"{remote_base}/{root_name}/{root_name}"
    try:
        result = subprocess.run(
            ["rclone", "lsf", "--dirs-only", search_path], 
            capture_output=True, text=True, check=True
        )
        dirs = [d.strip('/') for d in result.stdout.strip().split('\n') if d]
        if not dirs: return None
        return f"{root_name}/{root_name}/{dirs[0]}/images"
    except: return None

def run_pipeline():
    print("🚀 Biophysics Pipeline 多分流自動化啟動！\n" + "="*40)
    loader = DataLoader()
    base_output_dir = Path("./Processed_Data1")
    base_output_dir.mkdir(exist_ok=True)
    
    for root_folder in ROOT_FOLDERS:
        print(f"\n▶️ 處理專案: {root_folder}")
        temp_dir = None
        safe_name = root_folder
        
        # ---------------------------------------------------------
        # 💡 分流判斷邏輯
        # ---------------------------------------------------------
        # 空間分佈類 (只需少量採樣)
        is_spatial = any(x in root_folder for x in ["BCL", "half_half", "OFF"])
        # 時間動力學類 (需完整追蹤)
        is_dynamic = any(x in root_folder for x in ["switch", "ramp"])

        try:
            target_path = get_dynamic_image_path(root_folder)
            if not target_path: continue

            # ==========================================
            # 1. 資料載入 (根據分流決定下載策略)
            # ==========================================
            if is_spatial:
                print(f"  📍 [模式: 空間分析] 僅下載每 30 秒之影像...")
                temp_dir = loader.fetch_experiment_data(target_path, interval_seconds=30)
            else:
                print(f"  ⚡ [模式: 動力學分析] 啟動全量下載...")
                temp_dir = loader.fetch_experiment_data(target_path)
            
            # 統一進行數字排序
            raw_images = sorted(
                list(temp_dir.glob("fig_*.jpeg")), 
                key=lambda p: float(p.stem.replace("fig_", ""))
            )
            
            if not raw_images: continue
            print(f"  📦 取得檔案數: {len(raw_images)} 張")

            # ==========================================
            # 2. 數量守門員 (僅針對動力學分析)
            # ==========================================
            if is_dynamic and len(raw_images) >= 400:
                print(f"  ⏭️ 跳過：動力學案例圖片過多 ({len(raw_images)}張)")
                loader.cleanup(temp_dir)
                continue

            # ==========================================
            # 3. 轉灰階 (通用步驟)
            # ==========================================
            gray_folder = base_output_dir / f"Grayscale_{safe_name}"
            gray_image_paths = analysis.batch_convert_to_grayscale(raw_images, gray_folder)
            
            # ==========================================
            # 4. 微觀分析 (僅限動力學案例)
            # ==========================================
            # ==========================================
            if is_dynamic:
                print(f"  🧪 執行 Particle Tracking 與動力學分析...")
                kinematics_data, raw_trajectories = analysis.extract_trajectories_and_kinematics(gray_image_paths, fps=2.0)
                
                # 存 CSV 軌跡
                csv_file = base_output_dir / f"Trajectories_{safe_name}.csv"
                analysis.save_trajectories_to_csv(raw_trajectories, csv_file)

                # 存分析圖表 (強制關閉紅色光照陰影)
                plot_file = base_output_dir / f"Kinematics_Plot_{safe_name}.png"
                # 💡 直接傳入 light_on_range=None
                visualizer.plot_population_kinematics(kinematics_data, plot_file, light_on_range=None)

            # ==========================================
            # 5. 巨觀視覺化 (通用步驟)
            # ==========================================
            # 產出 2x5 時間序列畫布
            canvas_file = base_output_dir / f"Canvas_TimeSeries_{safe_name}.png"
            visualizer.create_time_series_canvas(gray_image_paths, canvas_file, max_images=10, cols=5)
            
            # 產出密度疊加圖 (取最後 1/3 做穩態)
            ss_count = max(1, len(gray_image_paths) // 3)
            density_matrix = analysis.calculate_density_map_fast(gray_image_paths)
            visualizer.save_density_image(density_matrix, base_output_dir / f"DensityMap_{safe_name}.png")
            
            print(f"✨ [{root_folder}] 處理結束！")

        except Exception as e:
            print(f"❌ 錯誤: {e}")
        finally:
            if temp_dir and temp_dir.exists():
                loader.cleanup(temp_dir)
                
    print("\n🎉 所有分流任務處理完畢！")

if __name__ == "__main__":
    run_pipeline()