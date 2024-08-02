import pyrealsense2 as rs
import cv2
import numpy as np

# 相机配置类
class RealsenseCamera:
    def __init__(self, camera_id_list, width, height, fps):
        self.camera_id_list = camera_id_list
        self.width = width
        self.height = height
        self.fps = fps
        self.pipelines = []
        self.configs = []
        self.aligns = []

    def camera_config(self):
        # 获取所有连接的RealSense设备的序列号
        connect_device = []
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs.camera_info.serial_number))
        
        # 检查是否连接了足够的相机
        if len(connect_device) < len(self.camera_id_list):
            print("需要连接两个RealSense相机，但当前只连接了{}个。".format(len(connect_device)))
            exit()

        # 为每个相机创建管道和配置
        for id in self.camera_id_list:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(connect_device[id])
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipelines.append(pipeline)
            self.configs.append(config)

        # 启动所有管道
        for pipeline, config in zip(self.pipelines, self.configs):
            pipeline.start(config)

    def get_frames(self):
        frames = []
        for pipeline in self.pipelines:
            frames.append(pipeline.wait_for_frames())

        return frames

    def rgb_image(self, frames, camera_id):
        color_frame = frames[camera_id].get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        print("相机退出成功。")

# 使用两个RealSense相机
camera_list = [0, 1]  # 假设有两个相机连接
camera = RealsenseCamera(camera_list, 640, 480, 30)
camera.camera_config()

try:
    while True:
        frames = camera.get_frames()
        for i, frame in enumerate(frames):
            img = camera.rgb_image(frames, i)
            cv2.imshow("RealSense Camera {}".format(i), img)
            print("FPS: {:.2f}".format(1.0 / (time.time() - start)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    camera.stop()
    cv2.destroyAllWindows()