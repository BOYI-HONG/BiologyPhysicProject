# main.py
import os
import json
import pandas as pd
import argparse  # <--- 新增這個
import config
import tracker
import plotter
import DisplayRender

def get_current_config():
    """自動提取 config.py 中所有大寫的設定變數，轉換為字典"""
    config_dict = {}
    for key in dir(config):
        if key.isupper():
            val = getattr(config, key)
            # 確保只儲存可序列化的基本資料型態
            if isinstance(val, (int, float, str, bool, type(None))):
                config_dict[key] = val
    return config_dict

def main():
    print("=== 眼蟲動力學分析管線啟動 ===")
    
    # --- 新增：解析命令列引數 ---
    parser = argparse.ArgumentParser(description="處理單一影片的分析")
    parser.add_argument('--video', type=str, help='指定要處理的影片路徑')
    parser.add_argument('--output', type=str, help='指定結果輸出的資料夾路徑')
    args = parser.parse_args()

    # 如果有從 Bash 傳入參數，就覆寫 config.py 裡面的預設值
    if args.video:
        config.VIDEO_FILE = args.video
    if args.output:
        config.OUTPUT_DIR = args.output
        
    # ==========================================
    # 下面的程式碼完全不用動！
    # 它們會自動使用覆寫過後的 config.VIDEO_FILE 和 config.OUTPUT_DIR
    # ==========================================
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"📁 輸出資料夾準備完畢: {config.OUTPUT_DIR}")
    
    # 定義檔案路徑
    csv_path = os.path.join(config.OUTPUT_DIR, 'tracking_data.csv')
    meta_path = os.path.join(config.OUTPUT_DIR, 'run_meta.json')
    video_output_path = os.path.join(config.OUTPUT_DIR, 'tracking_verification.mp4')
    
    # 取得當前的設定值
    current_config = get_current_config()
    
    # --- 判斷是否可以跳過追蹤 (Cache 機制) ---
    skip_tracking = False
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            # 比對設定檔是否完全相同
            if saved_config == current_config:
                skip_tracking = True
        except Exception as e:
            print(f"⚠️ 讀取設定檔紀錄失敗 ({e})，將重新執行追蹤。")

    # ==========================================
    # 階段 1: 影像追蹤與資料讀取
    # ==========================================
    if skip_tracking:
        print("\n--- 階段 1: 影像追蹤 (快取命中，跳過運算) ---")
        print(f"✅ 檢測到相同的設定檔，直接讀取既有數據: {csv_path}")
        final_data = pd.read_csv(csv_path)
    else:
        print("\n--- 階段 1: 影像追蹤 (全新運算) ---")
        raw_points = tracker.extract_positions(config.VIDEO_FILE, max_frames=config.MAX_FRAMES)
        
        if raw_points.empty:
            print("❌ 未抓取到資料，程式中止。")
            return
            
        tracks = tracker.link_data(raw_points)
        final_data = tracker.calculate_movement_angles(tracks)
        
        # 儲存軌跡數據
        final_data.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"✅ 軌跡數據已儲存: {csv_path}")
        
        # 儲存設定檔 Metadata
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, indent=4)
        print(f"✅ 設定檔狀態已備份: {meta_path}")

    # ==========================================
    # 階段 2: 影片渲染
    # ==========================================
    print("\n--- 階段 2: 追蹤驗證影片渲染 ---")
    if skip_tracking and os.path.exists(video_output_path):
        print(f"⏭️ 影片已存在且追蹤數據未改變，跳過影片渲染: {video_output_path}")
    else:
        # 假設影片是 30 FPS，tail_seconds=10 代表畫出前 300 幀的軌跡
        DisplayRender.render_tracking_video(config.VIDEO_FILE, final_data, video_output_path, tail_seconds=10)

    # ==========================================
    # 階段 3: 數據視覺化與物理分析
    # ==========================================
    print("\n--- 階段 3: 物理分析與繪圖 ---")
    # 背景噪音圖
    plotter.generate_background_image(config.VIDEO_FILE, config.OUTPUT_DIR, num_frames=config.MAX_FRAMES)
    # 動力學與群體分佈圖
    plotter.plot_kinematics(final_data, config.OUTPUT_DIR)
    plotter.plot_population_statistics(final_data, config.OUTPUT_DIR)
    plotter.plot_raw_scatter(final_data, config.OUTPUT_DIR)
    
    # 速度與角速度分析 (剛才新增的兩個函數)
    try:
        plotter.plot_speed_kinematics(final_data, config.OUTPUT_DIR)
        plotter.plot_speed_distributions(final_data, config.OUTPUT_DIR)
    except AttributeError:
        print("⚠️ 未檢測到速度分析函數，若不需要請忽略。")
        
    plotter.plot_population_time_series_with_shade(final_data, config.OUTPUT_DIR, light_span=config.LIGHT_SPAN)
    plotter.plot_mean_speed_scatter_with_marginals(final_data, config.OUTPUT_DIR)

    print("\n🎉 分析全部完成！請前往輸出資料夾查看結果。")

if __name__ == "__main__":
    main()