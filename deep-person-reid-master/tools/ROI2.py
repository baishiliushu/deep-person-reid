import os
import time
import xml.etree.ElementTree as ET
import cv2
from datetime import timedelta

import os

import os

def extract_extra_subpath_err(img_path, root_path, nfs_info=None):
    if not (img_path and root_path):
        return None

    # 处理 nfs_info 默认值和合法性
    if nfs_info is None:
        nfs_info = {}
    if nfs_info != {}:
        nl = nfs_info.get("nfs_local")
        lr = nfs_info.get("logic_root")
        if not (isinstance(nl, str) and nl.strip() and isinstance(lr, str) and lr.strip()):
            return None
        nfs_local = os.path.normpath(nl)
        logic_root = os.path.normpath(lr)
    else:
        nfs_local = logic_root = None

    # 获取目录并标准化
    img_dir = os.path.dirname(os.path.normpath(img_path))
    if not img_dir:
        return None

    # 应用 NFS 映射（如果启用且匹配）
    if nfs_local is not None:
        if img_dir.startswith(nfs_local):
            suffix = img_dir[len(nfs_local):]
            if suffix and not suffix.startswith(os.sep):
                return None  # 非完整目录边界（如 /abc vs /abcd）
            img_dir = logic_root + suffix

    root_norm = os.path.normpath(root_path)

    # 计算相对路径
    try:
        rel = os.path.relpath(img_dir, root_norm)
    except Exception:
        return None

    if rel == ".":  
        return "" # 完全相同
    if os.pardir in rel.split(os.sep):
        return None  # 不在子目录下

    # 统一正斜杠并加结尾 '/'
    return rel.replace(os.sep, "/") + "/"

def extract_extra_subpath(xml_img_path, local_img_path, nfs_info={}):
    """
    提取 xml_img_path 中相对于 local_img_path 多出的子目录路径（不含文件名），兼容 NFS 映射。

    Args:
        img_path (str): 图像完整路径
        root_path (str): 本地逻辑根路径
        nfs_info (dict or None): 可选，包含 'nfs_local' 和 'logic_root' 的映射

    Returns:
        str or None: 多出的子路径（如 "a/b/c/"），若无法计算则返回 None
    """
    sub_path = None
    if nfs_info is None:
        nfs_info = {}
    
    
    # 去掉文件名，只保留目录
    dir_path_all = os.path.dirname(xml_img_path)
    print("path in xml \n{}".format(xml_img_path, dir_path_all))
    dir_path = dir_path_all
    if nfs_info != {}:
        try:
            dir_path = str(dir_path_all).replace(nfs_info["nfs_local"], nfs_info["logic_root"])
            print("DEBUG \n{}".format(dir_path))
        except Exception: 
            return None
    sub_path = dir_path.replace(local_img_path+"/", "")
        
    print("sub_path is \n{}".format(sub_path))
    return sub_path

def extract_person_roi(xml_root, img_root, output_root, nfs_info={}):
    """
    递归提取所有子目录中的行人ROI，并按原目录结构保存（带进度提示）
    :param xml_root: XML文件根目录
    :param img_root: 图片根目录
    :param output_root: ROI保存根目录
    """
    # 第一步：先统计所有XML文件总数（用于进度计算）
    total_xml = 0
    xml_files = []  # 存储所有XML文件路径
    for dirpath, _, filenames in os.walk(xml_root):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_files.append(os.path.join(dirpath, filename))
                total_xml += 1

    if total_xml == 0:
        print("未找到任何XML文件，程序退出")
        return

    print(f"开始处理，共发现 {total_xml} 个XML文件...")
    start_time = time.time()  # 记录开始时间
    empty_file_count = 0
    # 第二步：遍历处理所有XML文件
    for count, xml_path in enumerate(xml_files, 1):
        if empty_file_count > 10:
            print("{}".format(nfs_info))
            print("check print or dir")
            exit(-1)
        # 计算当前进度和预估剩余时间
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / count  # 平均每个文件处理时间
        remaining_files = total_xml - count
        remaining_time = timedelta(seconds=int(avg_time_per_file * remaining_files))

        # 打印进度信息（覆盖当前行，保持界面整洁）
        progress = (count / total_xml) * 100
        print(
            f"\r进度: {count}/{total_xml} ({progress:.1f}%) | 已用时: {int(elapsed_time)}s | 预估剩余: {remaining_time}\n",
            end="")

        # 解析XML文件
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"\n解析XML失败 {xml_path}: {e}")
            continue

        # 获取图片文件名
        img_filename = root.find('filename').text
        if not img_filename:
            print(f"\nXML {os.path.basename(xml_path)} 中未找到文件名，跳过")
            continue

        img_path = root.find('path').text
        # 计算相对路径（匹配图片目录）
        relative_path = extract_extra_subpath(img_path,img_root,nfs_info) #os.path.relpath(os.path.dirname(xml_path), xml_root)
        print("relative_path is {}".format(relative_path))
        if relative_path is None:
            continue
        
        img_path = os.path.join(img_root, relative_path, img_filename)
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"\n图片不存在 {img_path}，跳过")
            empty_file_count = empty_file_count + 1
            continue
        else:
            print("{}".format(img_path))
        
        # 构建图片路径
        #img_dir = os.path.join(img_root, relative_path)
        #img_path = os.path.join(img_dir, img_filename)

        

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n无法读取图片 {img_path}，跳过")
            continue

        # 提取行人类别ROI
        person_count = 0
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name and cls_name.lower() == 'person':
                person_count += 1
                # 获取边界框
                bndbox = obj.find('bndbox')
                if not bndbox:
                    print(f"\nXML {os.path.basename(xml_path)} 中未找到边界框，跳过该目标")
                    continue

                # 解析坐标（支持浮点数转换）
                try:
                    xmin = int(round(float(bndbox.find('xmin').text)))
                    ymin = int(round(float(bndbox.find('ymin').text)))
                    xmax = int(round(float(bndbox.find('xmax').text)))
                    ymax = int(round(float(bndbox.find('ymax').text)))
                except (TypeError, ValueError) as e:
                    print(f"\nXML {os.path.basename(xml_path)} 坐标解析错误: {e}，跳过该目标")
                    continue

                # 确保坐标有效
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img.shape[1], xmax)
                ymax = min(img.shape[0], ymax)

                # 计算ROI长宽（关键判断依据）
                roi_width = xmax - xmin
                roi_height = ymax - ymin

                # 过滤长宽未同时大于50的ROI
                if (roi_width * roi_height <= 2500) or (min(roi_width, roi_height) < 36 ):
                    print(f"\nXML {os.path.basename(xml_path)} 中ROI尺寸过小（{roi_width}x{roi_height}），跳过该目标")
                    continue

                # 裁剪ROI
                roi = img[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    print(f"\nXML {os.path.basename(xml_path)} 中边界框无效，跳过该目标")
                    continue

                # 构建ROI保存路径（保持目录结构）
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # 提取路径中的层级信息用于命名
                rel_img_path = os.path.relpath(img_path, img_root)
                path_components = rel_img_path.split(os.sep)

                # 提取需要的层级信息
                level1 = path_components[0].split('-threeperson')[0] if len(path_components) > 0 else ''
                level2 = path_components[1].split('_')[-1] if len(path_components) > 1 else ''
                filename_body = os.path.splitext(path_components[-1])[0] if len(path_components) > 0 else ''

                # 组合生成新文件名
                img_ext = os.path.splitext(img_filename)[1]
                roi_filename = f"{level1}-{level2}-{filename_body}_person_{person_count}{img_ext}"

                # 保存ROI
                roi_path = os.path.join(output_dir, roi_filename)
                cv2.imwrite(roi_path, roi)

                # 每处理10个文件打印一次详细信息
                if count % 10 == 0:
                    print(f"\n已保存ROI: {roi_path}")

    # 处理完成，打印总耗时
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n\n所有文件处理完成！总耗时: {total_time}")
    print(f"ROI已按目录结构保存到: {output_root}")


if __name__ == "__main__":
    # 配置路径
    xml_root = "/home/leon/mount_point_d/test-result-moved/reid_datas/251104_track_1104-xml"#"/home/spring/nfs_222_testmoved/reid_datas/251020-dl-lxyu-log-datas/data/20251020-dl-i0-xml"
    img_root = "/home/leon/mount_point_d/test-result-moved/reid_datas/251104_track_1104"#"/home/spring/nfs_222_testmoved/reid_datas/251020-dl-lxyu-log-datas/data"
    output_root = "/home/leon/mount_point_d/test-result-moved/reid_datas/251104_track_1104_ROI"#"/home/spring/nfs_222_testmoved/reid_datas/251020-dl-lxyu-log-datas/data/20251020-dl-i0-ROI"
    
    nfs_info = {"nfs_local": "/home/spring/nfs_222_testmoved/", "logic_root": "/home/leon/mount_point_d/test-result-moved/"}
    # 执行提取
    extract_person_roi(xml_root, img_root, output_root, nfs_info)

    
