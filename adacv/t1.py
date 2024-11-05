import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'C:\Users\w\Desktop\adacv\test\p2.png')

# 转换为HSV色彩空间，用于颜色过滤
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义绿色的HSV范围（可以根据实际情况调整范围）
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# 应用掩膜
filtered_image = cv2.bitwise_and(image, image, mask=mask)

# 转换为灰度图
gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# 高斯模糊，平滑图像
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二值化
_, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

# 检测轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 将轮廓按y坐标排序
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# 计数变量
count = 0
detected_pods = []

# 遍历每个轮廓
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h  # 计算长宽比
    area = cv2.contourArea(contour)  # 计算轮廓面积
    perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

    # 根据长宽比、面积和圆形度过滤角果的轮廓
    if 0.1 < aspect_ratio < 0.9 and area > 200 and 0.1 < circularity < 0.3:

        merged = False


        # 尝试合并相邻的轮廓
        for j, existing in enumerate(detected_pods):
            ex, ey, ew, eh = existing
            # 如果两个轮廓在垂直方向上重叠且水平距离接近，合并为一个角果
            if abs(x - ex) < max(w, ew) and abs(y - (ey + eh)) < 20:
                # 更新现有角果的边界框
                nx = min(x, ex)
                ny = min(y, ey)
                nw = max(x + w, ex + ew) - nx
                nh = max(y + h, ey + eh) - ny
                detected_pods[j] = (nx, ny, nw, nh)
                merged = True
                break
        if not merged:
            detected_pods.append((x, y, w, h))
            count += 1
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # 绘制轮廓

# 绘制每个检测到的角果
for (x, y, w, h) in detected_pods:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 在图像右上角显示计数结果
cv2.putText(image, f"Count: {count}", (image.shape[1] - 150, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow("Counted Pods", image)
cv2.waitKey(0)
cv2.destroyAllWindows()