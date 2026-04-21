#!/bin/bash

# ==========================================
# 1. 定義資料集陣列 (對應好影片與輸出資料夾)
# ==========================================
# VIDEOS=(
# )

# OUTPUTS=(
# )

VIDEOS=(
    "/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_36/2023_06_26_Euglena_36.h264"
    "/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_26_Euglena_37/2023_06_26_Euglena_37.h264"
    "/mnt/j/BiologicalPhysics/V1/Euglena_circle_light/Euglena_circle_light/2023_06_15_Euglena_16/video.h264"
    "/mnt/j/BiologicalPhysics/V1/Euglena_grad_lateral/Euglena_grad_lateral/2023_06_15_Euglena_13/video.h264"
    "/mnt/j/BiologicalPhysics/V1/Euglena_grad_lateral/Euglena_grad_lateral/2023_06_26_Euglena_31/2023_06_26_Euglena_31.h264"
    "/mnt/j/BiologicalPhysics/V1/Euglena_grad_lateral/Euglena_grad_lateral/2023_06_26_Euglena_32/2023_06_26_Euglena_32.h264"
)

OUTPUTS=(
    "./Analysis_Results/Euglena_circle_light/2023_06_26_Run36"
    "./Analysis_Results/Euglena_circle_light/2023_06_26_Run37"
    "./Analysis_Results/Euglena_circle_light/2023_06_15_Run16"
    "./Analysis_Results/Euglena_grad_lateral/2023_06_15_Run13"
    "./Analysis_Results/Euglena_grad_lateral/2023_06_26_Run31"
    "./Analysis_Results/Euglena_grad_lateral/2023_06_26_Run32"
)
echo "=== 開始平行批次處理眼蟲影片 ==="

# 取得陣列的長度
num_jobs=${#VIDEOS[@]}

# ==========================================
# 2. 迴圈派發任務
# ==========================================
for (( i=0; i<$num_jobs; i++ )); do
    VIDEO_PATH="${VIDEOS[$i]}"
    OUT_PATH="${OUTPUTS[$i]}"
    
    echo "🚀 啟動進程 $[i+1]/$num_jobs : 輸出至 $OUT_PATH"
    
    # 執行 Python 主程式，傳入引數，並加上 & 讓它在背景平行執行
    # 這裡將終端機輸出導向到各資料夾的 log.txt，畫面才不會被多個進程的 print 洗版
    mkdir -p "$OUT_PATH"
    python main.py --video "$VIDEO_PATH" --output "$OUT_PATH" > "$OUT_PATH/terminal_log.txt" 2>&1 &
    
done

# ==========================================
# 3. 等待收割
# ==========================================
echo "⏳ 所有任務已丟入背景執行，等待完成中..."
wait # 這個指令會卡住，直到上面所有背景任務 (&) 都執行結束

echo "🎉 全部影片分析完畢！"