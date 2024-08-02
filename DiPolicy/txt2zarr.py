import os
import zarr
import numpy as np
import numcodecs

def parse_txt_to_zarr(calibration_folder):
    # 创建一个zarr组来存储数据
    root_group = zarr.open('replay_buffer.zarr', mode='w')

    # 在根组下创建"data"和"meta"子组
    data_group = root_group.create_group('data')
    meta_group = root_group.create_group('meta')

    # 初始化变量
    sequence_number = []
    timestamps = []
    stages = []
    actions = []
    eef_poses = []
    eef_poses_vel = []  # 保存末端执行器位置的速度
    joint_poses = []
    joint_poses_vel = []  # 保存关节位置的速度
    records_per_file = []  # 保存每个文件的记录数
    record_count = 0
    prev_timestamp = None
    prev_eef_pose = None
    prev_joint_pose = None

    # 遍历calibration文件夹中的所有文件夹
    for subdir, dirs, files in os.walk(calibration_folder):
        for file in files:
            if file == 'robot_data.txt':
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as file:
                    line_counter = 0  # 新增计数器
                    while True:
                        # 读取序列号，时间戳和stage
                        line = file.readline()
                        if not line:
                            break  # 如果没有更多的行，退出循环
                        if line_counter % 3 == 0:  # 每三条记录取一条
                            parts = line.split()
                            sequence_number.append(int(parts[0]))
                            current_timestamp = float(parts[1])
                            timestamps.append(current_timestamp)
                            stages.append(int(parts[2]))

                            # 读取action
                            action = file.readline().split()
                            actions.append([float(x) for x in action])

                            # 读取eef_pose
                            eef_pose = file.readline().split()
                            eef_pose = [float(x) for x in eef_pose]
                            eef_poses.append(eef_pose)

                            # 计算eef_pose速度
                            if prev_timestamp is not None and prev_eef_pose is not None:
                                time_diff = current_timestamp - prev_timestamp
                                eef_pose_vel = [(e - p) / time_diff for e, p in zip(eef_pose, prev_eef_pose)]
                            else:
                                eef_pose_vel = [0.0] * len(eef_pose)
                            eef_poses_vel.append(eef_pose_vel)

                            # 如果有joint_pose，读取它
                            joint_pose_line = file.readline()
                            if joint_pose_line.strip():  # 检查是否是空行
                                joint_pose = joint_pose_line.split()
                                joint_pose = [float(x) for x in joint_pose]
                                joint_poses.append(joint_pose)

                                # 计算joint_pose速度
                                if prev_timestamp is not None and prev_joint_pose is not None:
                                    joint_pose_vel = [(j - p) / time_diff for j, p in zip(joint_pose, prev_joint_pose)]
                                else:
                                    joint_pose_vel = [0.0] * len(joint_pose)
                                joint_poses_vel.append(joint_pose_vel)
                            else:
                                joint_poses.append([])  # 如果没有joint_pose，添加一个空列表
                                joint_poses_vel.append([])

                            prev_timestamp = current_timestamp
                            prev_eef_pose = eef_pose
                            prev_joint_pose = joint_pose if joint_pose_line.strip() else None

                            record_count += 1
                        else:
                            # 跳过接下来的三行
                            file.readline()  # action
                            file.readline()  # eef_pose
                            file.readline()  # joint_pose 或 空行
                        line_counter += 1
                    records_per_file.append(record_count)  # 保存当前文件的记录数
                    # 重置前一个记录的变量，以便为下一个文件计算速度
                    prev_timestamp = None
                    prev_eef_pose = None
                    prev_joint_pose = None

    # 将数据保存在"data"子组中
    data_group.array('timestamp', np.array(timestamps, dtype=float))
    data_group.array('stage', np.array(stages, dtype=int))
    data_group.array('action', np.array(actions, dtype=float))
    data_group.array('robot_eef_pose', np.array(eef_poses, dtype=float))
    data_group.array('robot_eef_pose_vel', np.array(eef_poses_vel, dtype=float))  # 保存末端执行器位置的速度
    # 使用Pickle编解码器来保存object类型的数组
    data_group.array('robot_joint', np.array(joint_poses, dtype=object), object_codec=numcodecs.Pickle())
    data_group.array('robot_joint_vel', np.array(joint_poses_vel, dtype=object), object_codec=numcodecs.Pickle())  # 保存关节位置的速度

    # 将records_per_file保存在"meta"子组中
    meta_group.array('episode_ends', np.array(records_per_file, dtype=int))  # 保存每个文件的记录数


if __name__ == '__main__':
    calibration_folder = '/home/yujiandong/diffusion_policy/newRobotTest/calibration'  # 替换为你的calibration文件夹路径
    parse_txt_to_zarr(calibration_folder)
    print('Data has been successfully saved to replay_buffer.zarr')