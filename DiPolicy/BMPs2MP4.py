import cv2
import os

def images_to_video(image_folder, output_path, fps, side):
    # 根据side参数筛选图片
    images = [img for img in os.listdir(image_folder) if img.endswith(".bmp") and side in img]
    if not images:  # 如果没有找到对应侧的图片，直接返回
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in sorted(images, key=lambda x: int(x.split('_')[0])):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()


def process_folder(datasets_folder, output_folder, fps):
    output_index = 0  # 输出文件夹编号

    for folder_name in os.listdir(datasets_folder):
        if os.path.isdir(os.path.join(datasets_folder, folder_name)):
            output_index_folder = os.path.join(output_folder, str(output_index))
            os.makedirs(output_index_folder, exist_ok=True)

            image_folder = os.path.join(datasets_folder, folder_name)

            # 分别为左侧和右侧图片生成视频
            output_path_left = os.path.join(output_index_folder, '3.mp4')
            output_path_right = os.path.join(output_index_folder, '1.mp4')

            images_to_video(image_folder, output_path_left, fps, 'left')
            images_to_video(image_folder, output_path_right, fps, 'right')

            output_index += 1


# 设置文件夹路径和帧率
datasets_folder = './calibration'  # 数据集文件夹路径
output_folder = './videos'  # 输出文件夹路径
fps = 30  # 帧率

# 处理文件夹中的所有图片
process_folder(datasets_folder, output_folder, fps)
