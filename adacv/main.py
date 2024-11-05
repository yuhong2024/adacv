import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'C:\Users\w\Desktop\adacv\test\p3.jpg')

# 转换为HSV色彩空间，用于颜色过滤
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义绿色的HSV范围
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

# 假设的比例尺值（例如每个像素代表0.1 cm）
scale = 0.1  # 每个像素的实际物理长度（单位：厘米）
count = 0
detected_pods = []

# 遍历每个轮廓
for contour in contours:
    # 计算轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(contour)
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

# 绘制并标注每个检测到的角果
for idx, (x, y, w, h) in enumerate(detected_pods, 1):
    # 绘制绿色外框
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顶部5%为细柄
    stem_start_y = y
    stem_end_y = y + int(h * 0.05)
    stem_length = (stem_end_y - stem_start_y) * scale
    cv2.rectangle(image, (x, stem_start_y), (x + w, stem_end_y), (255, 0, 255), 2)  # 紫色表示细柄
    cv2.putText(image, f"{stem_length:.1f} cm", (x - 50, (stem_start_y + stem_end_y) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # 中间90%为果实
    fruit_start_y = y + int(h * 0.05)
    fruit_end_y = y + int(h * 0.95)
    fruit_length = (fruit_end_y - fruit_start_y) * scale
    cv2.rectangle(image, (x, fruit_start_y), (x + w, fruit_end_y), (255, 0, 0), 2)  # 蓝色表示果实部分
    cv2.putText(image, f"{fruit_length:.1f} cm", (x - 50, (fruit_start_y + fruit_end_y) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 底部5%为尾部尖端
    tip_start_y = y + int(h * 0.95)
    tip_end_y = y + h
    tip_length = (tip_end_y - tip_start_y) * scale
    cv2.rectangle(image, (x, tip_start_y), (x + w, tip_end_y), (0, 255, 255), 2)  # 黄色表示尾部尖端
    cv2.putText(image, f"{tip_length:.1f} cm", (x - 50, (tip_start_y + tip_end_y) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 标注角果的计数序号
    cv2.putText(image, f"#{idx}", (x + w + 10, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 在图像左上角显示总计数结果
cv2.putText(image, f"Total Count: {count}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow("Counted Pods with Part Lengths", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
