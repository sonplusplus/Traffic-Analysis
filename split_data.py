import os
import shutil
import random


INPUT_DIR = "labels"    #path to dataset (after xanylabeling export)
OUTPUT_DIR = "data_sig" #path to save
TRAIN_RATIO = 0.8 

# list of classes (base on dataset)
CLASSES = ['signal']

def create_dirs():
    dirs = [
        f"{OUTPUT_DIR}/images/train",
        f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train",
        f"{OUTPUT_DIR}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f" Đã tạo xong cấu trúc thư mục tại: {OUTPUT_DIR}/")

def split_and_copy_data():
    all_files = os.listdir(INPUT_DIR)
    images = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    valid_data = []
    for img in images:
        txt_file = os.path.splitext(img)[0] + '.txt'
        if os.path.exists(os.path.join(INPUT_DIR, txt_file)):
            valid_data.append((img, txt_file))
            
    print(f"Tìm thấy {len(valid_data)} cặp ảnh/nhãn hợp lệ.")


    random.shuffle(valid_data)

    train_size = int(len(valid_data) * TRAIN_RATIO)
    train_data = valid_data[:train_size]
    val_data = valid_data[train_size:]

    print(f" Đang copy: {len(train_data)} file vào Train, {len(val_data)} file vào Val...")

    def copy_files(data_list, split_type):
        for img_name, txt_name in data_list:
            # Đường dẫn gốc
            src_img = os.path.join(INPUT_DIR, img_name)
            src_txt = os.path.join(INPUT_DIR, txt_name)
            
            # Đường dẫn đích
            dst_img = os.path.join(OUTPUT_DIR, f"images/{split_type}", img_name)
            dst_txt = os.path.join(OUTPUT_DIR, f"labels/{split_type}", txt_name)
            
            # Copy
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)

    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    print(" Đã copy xong toàn bộ dữ liệu!")

def create_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    yaml_content = f"""train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Đã tạo thành công file: {yaml_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Không tìm thấy thư mục đầu vào: {INPUT_DIR}")
    else:
        create_dirs()
        split_and_copy_data()
        create_yaml()
        print("\nDONE.")