import subprocess
import tempfile
import shutil
from pathlib import Path

class DataLoader:
    def __init__(self, remote_base_path="gdrive:Biophysics_projects"):
        self.remote_base = remote_base_path
        self.temp_base = Path(tempfile.gettempdir())

    def fetch_experiment_data(self, folder_path, interval_seconds=None):
        """下載特定實驗資料夾到暫存區，並支援定時採樣過濾"""
        safe_name = str(folder_path).replace("/", "_")
        temp_dir = self.temp_base / f"rclone_temp_{safe_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        remote_url = f"{self.remote_base}/{folder_path}"
        
        # 基礎 rclone 下載指令
        cmd = ["rclone", "copy", remote_url, str(temp_dir), "--transfers", "16"]
        
        # ====================================================
        # 🚀 核心優化：如果指定了採樣間隔，動態生成過濾名單
        # ====================================================
        if interval_seconds is not None:
            filter_file = self.temp_base / f"filter_{safe_name}.txt"
            print(f"🎯 [DataLoader] 啟用精準過濾！只抓取每 {interval_seconds} 秒的影像...")
            
            with open(filter_file, "w") as f:
                # 假設實驗最長不超過 3600 秒 (1小時)，寫入我們要的檔名規則
                for t in range(0, 300, interval_seconds):
                    # {t:02d} 會確保 0 變成 00，30 變成 30，120 依然是 120
                    f.write(f"+ fig_{float(t)}.0.jpeg\n")
                
                # 最後加上這行：排除其他所有不在名單上的檔案
                f.write("- *\n")
                
            # 將過濾規則檔塞進指令中
            cmd.extend(["--filter-from", str(filter_file)])
        # ====================================================

        print(f"📥 [DataLoader] 正在極速下載資料至 {temp_dir} ...")
        subprocess.run(cmd, capture_output=True, check=True)
        
        return temp_dir

    def cleanup(self, temp_dir):
        print(f"🧹 [DataLoader] 正在清理暫存空間...")
        shutil.rmtree(temp_dir, ignore_errors=True)