"""
批量评估 AI 内容鉴真网关的准确率。
从 Chameleon / face / progan 测试集抽样，调用 API 检测，计算指标。
"""
import os
import sys
import json
import time
import random
import urllib.request
import io
from collections import defaultdict

API_URL = "http://localhost:8000/api/v1/detect"
DATASET_ROOT = r"D:\python object\AIDE-main\dataset"

# 抽样配置
SAMPLES_PER_CATEGORY = 30  # 每个 real/fake 类别抽样数
PROGAN_CATEGORIES = ["airplane", "cat", "dog", "person", "car"]

random.seed(42)


def list_images(directory):
    """列出目录中的图片文件。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    if not os.path.exists(directory):
        return files
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(directory, f))
    return files


def detect_image(image_path):
    """调用 API 检测单张图片。"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    boundary = "----EvalBoundary" + str(int(time.time() * 1000))
    body = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + os.path.basename(image_path) + '"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + img_bytes + ("\r\n--" + boundary + "--\r\n").encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "label": "error", "fake_prob": -1}


def evaluate_dataset(name, real_dir, fake_dir, max_per_class):
    """评估一个数据集。"""
    print(f"\n{'='*60}")
    print(f"评估数据集: {name}")
    print(f"  real: {real_dir}")
    print(f"  fake: {fake_dir}")
    print(f"{'='*60}")

    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    real_files = real_files[:max_per_class]
    fake_files = fake_files[:max_per_class]

    print(f"  抽样: {len(real_files)} real + {len(fake_files)} fake")

    results = []
    all_per_model = defaultdict(lambda: {"real_correct": 0, "fake_correct": 0, "real_total": 0, "fake_total": 0})

    for i, (img_path, true_label) in enumerate(
        [(f, "real") for f in real_files] + [(f, "fake") for f in fake_files]
    ):
        data = detect_image(img_path)
        pred_label = data.get("label", "error")
        fake_prob = data.get("fake_prob", -1)

        # 二值化：uncertain 视为错误（或单独统计）
        correct = pred_label == true_label
        results.append({
            "path": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "fake_prob": fake_prob,
            "correct": correct,
        })

        # 统计各模型表现
        for pm in data.get("per_model", []):
            mid = pm["model_id"]
            mp = pm["fake_prob"]
            mlabel = "fake" if mp > 0.5 else "real"
            all_per_model[mid][f"{true_label}_total"] += 1
            if mlabel == true_label:
                all_per_model[mid][f"{true_label}_correct"] += 1

        if (i + 1) % 20 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  进度: {i+1}/{len(real_files)+len(fake_files)}, 当前准确率: {acc:.2%}")

    # 计算指标
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total > 0 else 0

    # 混淆矩阵
    tp = sum(1 for r in results if r["true_label"] == "fake" and r["pred_label"] == "fake")
    fp = sum(1 for r in results if r["true_label"] == "real" and r["pred_label"] == "fake")
    tn = sum(1 for r in results if r["true_label"] == "real" and r["pred_label"] == "real")
    fn = sum(1 for r in results if r["true_label"] == "fake" and r["pred_label"] == "real")
    uncertain = sum(1 for r in results if r["pred_label"] == "uncertain")
    errors = sum(1 for r in results if r["pred_label"] == "error")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  --- {name} 结果 ---")
    print(f"  总数: {total}, 正确: {correct}, 准确率: {accuracy:.2%}")
    print(f"  混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  不确定(uncertain): {uncertain}, 错误(error): {errors}")
    print(f"  精确率(Precision): {precision:.2%}")
    print(f"  召回率(Recall): {recall:.2%}")
    print(f"  F1: {f1:.2%}")

    print(f"\n  --- 各模型单独准确率 ---")
    for mid, stats in sorted(all_per_model.items()):
        rt = stats["real_total"]
        ft = stats["fake_total"]
        rc = stats["real_correct"]
        fc = stats["fake_correct"]
        acc = (rc + fc) / (rt + ft) if (rt + ft) > 0 else 0
        print(f"    {mid:20s}: 总准确率={acc:.2%} (real={rc}/{rt}, fake={fc}/{ft})")

    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "uncertain": uncertain, "errors": errors,
        "precision": precision, "recall": recall, "f1": f1,
        "per_model": dict(all_per_model),
        "results": results,
    }


def main():
    print("AI 内容鉴真网关 - 批量准确率评估")
    print(f"API: {API_URL}")

    # 等待服务就绪
    for _ in range(10):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            print("服务就绪")
            break
        except:
            print("等待服务启动...")
            time.sleep(3)

    all_results = []

    # 1. Chameleon
    all_results.append(evaluate_dataset(
        "Chameleon",
        os.path.join(DATASET_ROOT, "Chameleon", "test", "0_real"),
        os.path.join(DATASET_ROOT, "Chameleon", "test", "1_fake"),
        SAMPLES_PER_CATEGORY,
    ))

    # 2. face
    all_results.append(evaluate_dataset(
        "face",
        os.path.join(DATASET_ROOT, "face", "test", "face", "0_real"),
        os.path.join(DATASET_ROOT, "face", "test", "face", "1_fake"),
        SAMPLES_PER_CATEGORY,
    ))

    # 3. progan (多个类别合并)
    progan_real = []
    progan_fake = []
    for cat in PROGAN_CATEGORIES:
        progan_real.extend(list_images(os.path.join(DATASET_ROOT, "progan", "test", cat, "0_real")))
        progan_fake.extend(list_images(os.path.join(DATASET_ROOT, "progan", "test", cat, "1_fake")))
    random.shuffle(progan_real)
    random.shuffle(progan_fake)
    # 每个类别 SAMPLES_PER_CATEGORY，总共 5*SAMPLES_PER_CATEGORY
    progan_real = progan_real[: SAMPLES_PER_CATEGORY * len(PROGAN_CATEGORIES)]
    progan_fake = progan_fake[: SAMPLES_PER_CATEGORY * len(PROGAN_CATEGORIES)]

    # 写入临时目录供评估（使用 D 盘，避免 C 盘空间不足）
    import shutil
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_eval")
    tmp_real = os.path.join(tmp_dir, "real")
    tmp_fake = os.path.join(tmp_dir, "fake")
    os.makedirs(tmp_real, exist_ok=True)
    os.makedirs(tmp_fake, exist_ok=True)
    for f in progan_real:
        shutil.copy(f, os.path.join(tmp_real, os.path.basename(f)))
    for f in progan_fake:
        shutil.copy(f, os.path.join(tmp_fake, os.path.basename(f)))

    all_results.append(evaluate_dataset(
        f"progan ({','.join(PROGAN_CATEGORIES)})",
        tmp_real, tmp_fake,
        SAMPLES_PER_CATEGORY * len(PROGAN_CATEGORIES),
    ))

    # 汇总
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    total_all = sum(r["total"] for r in all_results)
    correct_all = sum(r["correct"] for r in all_results)
    print(f"总样本: {total_all}, 总正确: {correct_all}, 总准确率: {correct_all/total_all:.2%}")
    for r in all_results:
        print(f"  {r['name']:30s}: {r['accuracy']:.2%} ({r['correct']}/{r['total']})")

    # 保存详细结果
    output_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
