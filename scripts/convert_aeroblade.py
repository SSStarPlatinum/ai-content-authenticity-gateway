"""
下载 Stable Diffusion VAE 并转换为 ONNX 格式，用于 AeroBlade 重建误差检测。
AeroBlade 原理：AI 生成图像通过 VAE 编码-解码后的重建误差低于真实图像。
"""
import os
import sys
import subprocess

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def install_deps():
    """安装必要依赖。"""
    print("[依赖] 检查/安装 diffusers, transformers, accelerate...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "diffusers", "transformers", "accelerate", "-q"
    ])
    print("  依赖安装完成")


def convert_vae():
    """下载 SD VAE 并导出编码器/解码器 ONNX。"""
    import torch
    import numpy as np
    import onnxruntime as ort
    from diffusers import AutoencoderKL

    print("\n" + "=" * 60)
    print("Stable Diffusion VAE -> ONNX")
    print("=" * 60)

    # 加载 VAE（使用 SD 1.5 的 VAE）
    print("[下载/加载] Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.eval()
    print("  VAE 加载成功")

    # 导出编码器
    print("[导出] VAE Encoder ONNX...")
    encoder_path = os.path.join(WEIGHTS_DIR, "aeroblade_encoder.onnx")

    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            # 编码，返回潜空间均值（跳过采样）
            h = self.vae.encoder(x)
            moments = self.vae.quant_conv(h)
            mean, _ = torch.chunk(moments, 2, dim=1)
            return mean

    encoder = VAEEncoderWrapper(vae)
    encoder.eval()

    dummy = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        encoder, dummy, encoder_path,
        opset_version=17,
        input_names=["input"], output_names=["latent"],
        dynamic_axes={"input": {0: "batch"}, "latent": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  编码器导出成功: {os.path.getsize(encoder_path)/1024/1024:.2f} MB")

    # 导出解码器
    print("[导出] VAE Decoder ONNX...")
    decoder_path = os.path.join(WEIGHTS_DIR, "aeroblade_decoder.onnx")

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z):
            # 解码潜空间
            z = self.vae.post_quant_conv(z)
            image = self.vae.decoder(z)
            return image

    decoder = VAEDecoderWrapper(vae)
    decoder.eval()

    dummy_latent = torch.randn(1, 4, 16, 16)
    torch.onnx.export(
        decoder, dummy_latent, decoder_path,
        opset_version=17,
        input_names=["latent"], output_names=["output"],
        dynamic_axes={"latent": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  解码器导出成功: {os.path.getsize(decoder_path)/1024/1024:.2f} MB")

    # 验证端到端推理
    print("[验证] 编码-解码端到端推理...")
    sess_enc = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    sess_dec = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

    test_img = np.random.randn(1, 3, 128, 128).astype(np.float32)
    latent = sess_enc.run(None, {"input": test_img})[0]
    recon = sess_dec.run(None, {"latent": latent})[0]
    mse = np.mean((test_img - recon) ** 2)
    print(f"  128x128: 输入{test_img.shape} -> 潜空间{latent.shape} -> 重建{recon.shape}, MSE={mse:.6f}")
    print("  AeroBlade VAE (128x128) 验证通过!")

    return True


if __name__ == "__main__":
    print("AeroBlade VAE 下载与 ONNX 转换")
    print(f"输出目录: {WEIGHTS_DIR}")

    try:
        install_deps()
        convert_vae()
        print("\n完成! aeroblade_encoder.onnx 和 aeroblade_decoder.onnx 已生成")
    except Exception as e:
        print(f"\n失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
