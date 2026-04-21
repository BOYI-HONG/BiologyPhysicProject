# config.py

# ==========================================
# 基本環境設定
# ==========================================
# 影片來源路徑
VIDEO_FILE = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_36/2023_06_26_Euglena_36.h264'
VIDEO_FILE = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_37/2023_06_26_Euglena_37.h264'
VIDEO_FILE = '/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_15_Euglena_16/video.h264'

# 輸出資料夾路徑 (主程式會自動建立)
OUTPUT_DIR = './Analysis_Results/Euglena_circle_light/2023_06_26_Run36'
OUTPUT_DIR = './Analysis_Results/Euglena_circle_light/2023_06_26_Run37'
OUTPUT_DIR = './Analysis_Results/Euglena_circle_light/2023_06_15_Run16'

# 處理設定
MAX_FRAMES = -1  # 設為 -1 則分析整部影片

# ==========================================
# 追蹤與影像處理參數設定區 (可依實驗微調)
# ==========================================
AREA_MIN = 30        # ROI 最小面積
AREA_MAX = 1000      # ROI 最大面積
SEARCH_RANGE = 30    # Trackpy 尋找下一幀的最大像素距離
MEMORY = 5           # Trackpy 容許眼蟲短暫消失的幀數
THRESHOLD = 10       # Trackpy 最短有效軌跡長度


# LIGHT_SPAN = (100, 300)
LIGHT_SPAN = None