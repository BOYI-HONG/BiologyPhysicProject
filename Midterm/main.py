# main.py
import os
import config
import tracker
import plotter

def main():
    print("=== 眼蟲動力學分析管線啟動 ===")
    
    # 1. 建立輸出環境
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"📁 輸出資料夾準備完畢: {config.OUTPUT_DIR}")

    # 2. 影像處理與軌跡提取
    print("\n--- 階段 1: 影像追蹤 ---")
    raw_points = tracker.extract_positions(config.VIDEO_FILE, max_frames=config.MAX_FRAMES)
    
    if raw_points.empty:
        print("❌ 未抓取到資料，程式中止。")
        return

    tracks = tracker.link_data(raw_points)
    final_data = tracker.calculate_movement_angles(tracks)
    
    # 儲存軌跡數據
    csv_path = os.path.join(config.OUTPUT_DIR, 'tracking_data.csv')
    final_data.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"✅ 軌跡數據已儲存: {csv_path}")

    # 3. 數據視覺化
    print("\n--- 階段 2: 物理分析與繪圖 ---")
    plotter.generate_background_image(config.VIDEO_FILE, config.OUTPUT_DIR, num_frames=config.MAX_FRAMES)
    plotter.plot_kinematics(final_data, config.OUTPUT_DIR)
    plotter.plot_population_statistics(final_data, config.OUTPUT_DIR)
    plotter.plot_raw_scatter(final_data, config.OUTPUT_DIR)

    print("\n🎉 分析全部完成！")

if __name__ == "__main__":
    main()