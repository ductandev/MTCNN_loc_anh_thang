import os
import shutil

path = "./data/"
files = os.listdir(path)

for names_folder in files:
    if not names_folder.endswith('.zip'):
        new_path = path + names_folder                      # new_path = './data/0079'
        # print(new_path)
        go_to_file = os.path.join(new_path)                 # go_to_file = './data/0079'
        # print(go_to_file)
        name_img_file = os.listdir(go_to_file)              # name_img_file = [Image1.png,Image2.png,.....Image100.png]
        # print(name_img_file)
        for names_img in name_img_file:
            path_to_img = go_to_file + "/" + names_img      # path_to_img = './data/1519/Image99.png' || đường dẫn tới files ảnh
            # print(path_to_img)
