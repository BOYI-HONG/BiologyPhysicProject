import cv2
import numpy as np
import os

def run_segmentation_pipeline(video_path, output_dir, num_frames=20):
    print(f"=== 開始執行影像分割練習 (前 {num_frames} 幀) ===")
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return

    # 取得原始影片屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # 預設防呆

    # 設定影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 準備五個 VideoWriter 來儲存每一步的結果
    out_1_gray   = cv2.VideoWriter(os.path.join(output_dir, 'step1_gray_8bit.mp4'), fourcc, fps, (width, height), isColor=False)
    out_2_otsu   = cv2.VideoWriter(os.path.join(output_dir, 'step2_otsu_thresh.mp4'), fourcc, fps, (width, height), isColor=False)
    out_3_morph  = cv2.VideoWriter(os.path.join(output_dir, 'step3_morph_closing.mp4'), fourcc, fps, (width, height), isColor=False)
    out_4_result = cv2.VideoWriter(os.path.join(output_dir, 'step4_final_roi.mp4'), fourcc, fps, (width, height), isColor=True)

    # 準備一個文字檔案來記錄每一幀的 Otsu 閾值和 ROI 資訊
    log_file = open(os.path.join(output_dir, 'segmentation_log.txt'), 'w', encoding='utf-8')

    frame_idx = 0
    # 把 while 條件改成這樣：
    while cap.isOpened():
        # 如果 num_frames 不是 -1，且已經達到指定幀數，就中斷迴圈
        if num_frames != -1 and frame_idx >= num_frames:
            break
            
        ret, frame = cap.read()
        if not ret: break
        
        # ==========================================
        # Step 1: 讀取原始影像並降為 8-bit 灰階
        # ==========================================
        # cv2.cvtColor 轉 GRAY 預設就是 8-bit (0-255) 的 np.uint8
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_1_gray.write(gray)

        # (選擇性) 為了讓 Otsu 效果更好，通常會先做一點輕微的模糊來去雜訊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ==========================================
        # Step 2 & 4: 計算直方圖並使用 Otsu 方法進行 Threshold
        # ==========================================
        # cv2.THRESH_OTSU 會自動計算最佳閾值，取代手動設定的固定數字
        # 因為眼蟲是亮的，背景是暗的，所以不用 THRESH_BINARY_INV
        otsu_thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out_2_otsu.write(thresh)

        # ==========================================
        # Step 5: 形態學處理 (Morphological Operations)
        # 解決「可能被切碎的眼蟲」
        # ==========================================
        # 定義結構元素 (Kernel)，這裡用 5x5 的橢圓形最適合生物體
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 使用 Closing (閉運算)：先膨脹後侵蝕
        # 這能把眼蟲內部因為失焦而產生的「黑洞」填滿，把碎成兩半的眼蟲接起來
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        out_3_morph.write(morph)

        # ==========================================
        # Step 6: 標記 ROI 位置以及面積
        # ==========================================
        # 尋找輪廓 (基於形態學處理後的乾淨二值圖)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 準備一張彩色圖來畫結果 (可以拿原始彩色 frame 來畫，比較清楚)
        result_frame = frame.copy()
        
        log_file.write(f"--- Frame {frame_idx} (Otsu Threshold: {otsu_thresh_val}) ---\n")
        
        roi_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 簡單過濾掉太小或太大的雜訊
            if 30 < area < 1000:
                roi_count += 1
                # 取得包圍框 (Bounding Box)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 畫綠色框框
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 標示面積數值
                cv2.putText(result_frame, f"A:{int(area)}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 寫入 Log
                log_file.write(f"  ROI {roi_count}: x={x}, y={y}, area={area}\n")

        # 標示總結資訊
        cv2.putText(result_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Otsu Thr: {otsu_thresh_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        cv2.putText(result_frame, f"Total ROIs: {roi_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        
        out_4_result.write(result_frame)
        
        print(f"已處理幀數: {frame_idx} / {num_frames} (Otsu: {otsu_thresh_val}, ROIs: {roi_count})")
        frame_idx += 1

    # 釋放資源
    cap.release()
    out_1_gray.release()
    out_2_otsu.release()
    out_3_morph.release()
    out_4_result.release()
    log_file.close()
    
    print(f"\n🎉 練習完成！請前往 {output_dir} 查看 4 支影片與 Log 檔。")

# (保留給單獨測試用)
if __name__ == "__main__":
    test_vid = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_36/2023_06_26_Euglena_36.h264'
    run_segmentation_pipeline(test_vid, './Practice_Results', -1)