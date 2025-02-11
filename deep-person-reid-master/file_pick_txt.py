file1_path = '/home/spring/test_qzj/data/try/duke_pick.txt'  ##file1 = pick
file2_path = '/home/spring/test_qzj/data/try/bounding_box_train.txt'## file2 = all

output_path = '/home/spring/test_qzj/data/try/filtered_bounding_box_train.txt'
# 读取第一个文件并提取文件名
with open(file1_path, 'r') as file1:
    lines1 = [line.strip().split('/')[-1] for line in file1.readlines()]

# 读取第二个文件
with open(file2_path, 'r') as file2:
    lines2 = file2.readlines()

# 遍历lines2并移除与lines1有相同文件名的行
filtered_lines2 = [line for line in lines2 if line.strip().split(',')[-3].split('/')[-1] not in lines1]

# 将结果写入新的文件

with open(output_path, 'w') as output_file:
    output_file.writelines(filtered_lines2)

print("过滤完成，结果已保存到", output_path)
