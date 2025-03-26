import skimage as sk
from skimage import io, color, restoration, util, metrics
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import gaussian_filter

def load_dataset(dataset_path):
    """加载数据集中的模糊图像和对应的清晰图像"""
    blurry_images = []
    sharp_images = []
    
    # 根据您的数据集结构调整此部分
    blurry_dir = os.path.join(dataset_path, 'blurry')
    sharp_dir = os.path.join(dataset_path, 'sharp')
    
    for filename in os.listdir(blurry_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            blurry_img = io.imread(os.path.join(blurry_dir, filename))
            # 假设清晰图像和模糊图像文件名相同
            sharp_img = io.imread(os.path.join(sharp_dir, filename))
            
            # 转换为灰度图像进行处理
            if blurry_img.ndim == 3:
                blurry_img = color.rgb2gray(blurry_img)
            if sharp_img.ndim == 3:
                sharp_img = color.rgb2gray(sharp_img)
                
            blurry_images.append(blurry_img)
            sharp_images.append(sharp_img)
    
    return blurry_images, sharp_images

def wiener_deconvolution(image, kernel, K=0.01):
    """维纳滤波去卷积"""
    # 进行傅里叶变换
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    
    # 维纳滤波公式
    kernel_fft_conj = np.conj(kernel_fft)
    restored = image_fft * kernel_fft_conj / (np.abs(kernel_fft)**2 + K)
    
    # 逆傅里叶变换
    restored_img = np.abs(np.fft.ifft2(restored))
    return restored_img

def richardson_lucy_deconvolution(image, psf, num_iter=30):
    """Richardson-Lucy去卷积算法"""
    return restoration.richardson_lucy(image, psf, num_iter)

def estimate_motion_blur_kernel(length=15, angle=45):
    """估计运动模糊核"""
    # 创建运动模糊核
    kernel = np.zeros((length, length))
    center = length // 2
    
    # 计算角度的弧度值
    rad = np.deg2rad(angle)
    
    # 在核中创建一条线
    for i in range(length):
        x_offset = int(np.round((i - center) * np.cos(rad)))
        y_offset = int(np.round((i - center) * np.sin(rad)))
        
        if 0 <= center + x_offset < length and 0 <= center + y_offset < length:
            kernel[center + y_offset, center + x_offset] = 1
    
    # 归一化核
    kernel = kernel / np.sum(kernel)
    return kernel

def evaluate_restoration(original, restored):
    """评估复原效果"""
    # 计算PSNR
    psnr = metrics.peak_signal_noise_ratio(original, restored)
    # 计算SSIM
    ssim = metrics.structural_similarity(original, restored)
    return psnr, ssim

def blind_deconvolution(image, psf_size=15, iterations=10):
    """盲去卷积"""
    # 使用OpenCV的盲去卷积
    # 注意：这只是一个简化版本，实际应用中可能需要更复杂的实现
    estimated_psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)
    restored = cv2.edgetaper(image, estimated_psf)
    
    for _ in range(iterations):
        # 使用当前估计的PSF进行去卷积
        restored = wiener_deconvolution(image, estimated_psf)
        # 更新PSF估计
        # 这里简化处理，实际中需要更复杂的PSF估计算法
        estimated_psf = estimate_motion_blur_kernel(psf_size, np.random.randint(0, 180))
    
    return restored

def main():
    # 设置数据集路径
    dataset_path = r"e:\作业\大三下\智能信息处理实践\BlurRestoration\dataset"
    
    # 加载数据集
    blurry_images, sharp_images = load_dataset(dataset_path)
    
    if not blurry_images:
        print("未找到图像，请检查数据集路径")
        return
    
    # 选择一张图像进行实验
    idx = 0
    blurry_img = blurry_images[idx]
    sharp_img = sharp_images[idx]
    
    # 1. 估计模糊核
    # 方法1：假设运动模糊，估计模糊核
    psf = estimate_motion_blur_kernel(15, 45)
    
    # 2. 使用不同方法进行图像复原
    # 方法1：维纳滤波
    wiener_restored = wiener_deconvolution(blurry_img, psf)
    
    # 方法2：Richardson-Lucy算法
    rl_restored = richardson_lucy_deconvolution(blurry_img, psf)
    
    # 方法3：盲去卷积（不需要预先知道PSF）
    blind_restored = blind_deconvolution(blurry_img)
    
    # 3. 评估复原效果
    wiener_psnr, wiener_ssim = evaluate_restoration(sharp_img, wiener_restored)
    rl_psnr, rl_ssim = evaluate_restoration(sharp_img, rl_restored)
    blind_psnr, blind_ssim = evaluate_restoration(sharp_img, blind_restored)
    
    print(f"维纳滤波: PSNR = {wiener_psnr:.2f}, SSIM = {wiener_ssim:.4f}")
    print(f"Richardson-Lucy: PSNR = {rl_psnr:.2f}, SSIM = {rl_ssim:.4f}")
    print(f"盲去卷积: PSNR = {blind_psnr:.2f}, SSIM = {blind_ssim:.4f}")
    
    # 4. 可视化结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(blurry_img, cmap='gray')
    plt.title('模糊图像')
    
    plt.subplot(2, 3, 2)
    plt.imshow(sharp_img, cmap='gray')
    plt.title('原始清晰图像')
    
    plt.subplot(2, 3, 3)
    plt.imshow(psf, cmap='gray')
    plt.title('估计的PSF')
    
    plt.subplot(2, 3, 4)
    plt.imshow(wiener_restored, cmap='gray')
    plt.title(f'维纳滤波 (PSNR: {wiener_psnr:.2f})')
    
    plt.subplot(2, 3, 5)
    plt.imshow(rl_restored, cmap='gray')
    plt.title(f'Richardson-Lucy (PSNR: {rl_psnr:.2f})')
    
    plt.subplot(2, 3, 6)
    plt.imshow(blind_restored, cmap='gray')
    plt.title(f'盲去卷积 (PSNR: {blind_psnr:.2f})')
    
    plt.tight_layout()
    plt.savefig(r"e:\作业\大三下\智能信息处理实践\BlurRestoration\results\restoration_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()