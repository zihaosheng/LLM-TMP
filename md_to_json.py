import os
import json
import re

def ends_with_number_in_parentheses(s):
    return bool(re.search(r" \(\d+\)$", s))  # 匹配 " (数字)" 结尾

# 设定主目录路径
base_dir = "Data_2025_2_25/Data(2025_2_25)"

# 获取所有不带 (1) 和 (2) 标号的主文件夹
main_folders = set()
for folder in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, folder)):
        if ends_with_number_in_parentheses(folder):
            continue
        base_name = folder.split("/")[-1]
        main_folders.add(base_name)

# 解析所有符合要求的文件夹
data_list = []
for main_folder in main_folders:
    current_folder_path = os.path.join(base_dir, main_folder)
    initial_folder_path = os.path.join(base_dir, main_folder+" (1)")

    if not os.path.exists(current_folder_path) or not os.path.isdir(current_folder_path):
        continue  # 确保路径存在且是文件夹
    if not os.path.exists(initial_folder_path) or not os.path.isdir(initial_folder_path):
        continue  # 确保路径存在且是文件夹

    # 获取 .md 文件
    current_md_files = [f for f in os.listdir(current_folder_path) if f.endswith(".md")]
    initial_md_files = [f for f in os.listdir(initial_folder_path) if f.endswith(".md")]

    if len(current_md_files) != 2 or len(initial_md_files) != 1:
        continue  # 只处理 md 文件正好有两个的文件夹

    # 读取两个 .md 文件
    current_md_files.sort()  # 排序以确保一致性，原文件排前，提取内容排后
    current_md_file, comment_md_file = current_md_files

    with open(os.path.join(current_folder_path, current_md_file), "r", encoding="utf-8") as f:
        current_tmp = f.read().strip()

    with open(os.path.join(current_folder_path, comment_md_file), "r", encoding="utf-8") as f:
        comments = f.read().strip()

    with open(os.path.join(initial_folder_path, initial_md_files[0]), "r", encoding="utf-8") as f:
        initial_tmp = f.read().strip()

    # 存入 JSON 数据
    data_list.append({
        "instruction": "[TMP comments]",
        "input": initial_tmp+current_tmp,
        "output": comments
    })

# 保存 JSON 文件
output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)

print(f"数据已保存到 {output_file}")
