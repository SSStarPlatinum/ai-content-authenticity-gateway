"""测量各模型单图推理时间。"""
import os, sys, time
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.registry import registry, auto_discover

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")

def main():
    auto_discover()
    # 创建测试图
    img = Image.new('RGB', (800, 600), color=(128, 128, 128))

    models_to_test = [
        "freq_detector", "ela_detector", "noise_detector",
        "cfa_detector", "color_detector",
        "cnn_spotter", "universal_fake", "aeroblade"
    ]

    print("各模型单图推理时间（取5次平均）：")
    print("-" * 50)

    for mid in models_to_test:
        try:
            model = registry.create(mid, weights_dir=WEIGHTS_DIR)
            model.load()

            # 预热
            model.predict(img)

            # 测量5次
            times = []
            for _ in range(5):
                start = time.perf_counter()
                model.predict(img)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg = np.mean(times)
            print(f"  {mid:20s}: {avg:8.1f} ms  (min={min(times):.0f}, max={max(times):.0f})")
        except Exception as e:
            print(f"  {mid:20s}: 失败 - {e}")

if __name__ == "__main__":
    main()
