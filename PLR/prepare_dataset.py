import os
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
import random

class LicensePlateDataset(Dataset):
    """Dataset cho ảnh biển số xe"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def extract_label_from_filename(filename):
    """Trích xuất tên biển số từ tên file"""
    # Lấy phần trước dấu "_" và loại bỏ extension
    base_name = filename.split('.')[0]
    license_plate = base_name.split('_')[0]
    return license_plate

def load_images_from_folder(folder_path):
    """Load ảnh từ folder và trích xuất label"""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} không tồn tại!")
        return images, labels
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in valid_extensions:
            img_path = os.path.join(folder_path, filename)
            try:
                # Load ảnh
                img = Image.open(img_path).convert('RGB')
                # Trích xuất label
                label = extract_label_from_filename(filename)
                
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Lỗi khi load {filename}: {e}")
    
    return images, labels

def create_dataset():
    """Tạo dataset từ các folder và lưu vào processed_dataset"""
    
    print("Đang load dữ liệu từ các folder...")
    
    # Load dữ liệu train từ 2 folders
    train_folder_1 = './dataset/synthetic_dataset_1/image_train'
    train_folder_2 = './dataset/image_train_augmented'
    
    images_1, labels_1 = load_images_from_folder(train_folder_1)
    images_2, labels_2 = load_images_from_folder(train_folder_2)
    
    # Kết hợp 2 datasets train
    X_train = images_1 + images_2
    y_train = labels_1 + labels_2
    
    print(f"Đã load {len(images_1)} ảnh từ {train_folder_1}")
    print(f"Đã load {len(images_2)} ảnh từ {train_folder_2}")
    print(f"Tổng số ảnh train: {len(X_train)}")
    
    # Shuffle dữ liệu train
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train, y_train = zip(*combined)
    X_train = list(X_train)
    y_train = list(y_train)
    
    # Load dữ liệu validation
    val_folder = './dataset/synthetic_dataset_1/val'
    X_val, y_val = load_images_from_folder(val_folder)
    
    print(f"Đã load {len(X_val)} ảnh validation từ {val_folder}")
    
    # Tạo folder output nếu chưa tồn tại
    output_folder = './dataset/processed_dataset'
    os.makedirs(output_folder, exist_ok=True)
    
    # Lưu datasets
    print("\nĐang lưu processed datasets...")
    
    with open(os.path.join(output_folder, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(os.path.join(output_folder, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(os.path.join(output_folder, 'X_val.pkl'), 'wb') as f:
        pickle.dump(X_val, f)
    
    with open(os.path.join(output_folder, 'y_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    
    print(f"Đã lưu datasets vào {output_folder}")
    
    # In thống kê
    print("\n=== Thống kê dataset ===")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Số lượng biển số unique (train): {len(set(y_train))}")
    print(f"Số lượng biển số unique (val): {len(set(y_val))}")
    print(f"Ví dụ labels train: {y_train[:5]}")
    print(f"Ví dụ labels val: {y_val[:5]}")
    
    return X_train, y_train, X_val, y_val

def load_processed_dataset(processed_folder='./dataset/processed_dataset'):
    """Load dataset đã xử lý từ file"""
    print(f"Đang load processed dataset từ {processed_folder}...")
    
    with open(os.path.join(processed_folder, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    
    with open(os.path.join(processed_folder, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    
    with open(os.path.join(processed_folder, 'X_val.pkl'), 'rb') as f:
        X_val = pickle.load(f)
    
    with open(os.path.join(processed_folder, 'y_val.pkl'), 'rb') as f:
        y_val = pickle.load(f)
    
    print("Đã load xong!")
    return X_train, y_train, X_val, y_val



# Ví dụ sử dụng
if __name__ == "__main__":
    # Bước 1: Tạo và lưu processed dataset
    print("=" * 50)
    print("BƯỚC 1: Tạo processed dataset")
    print("=" * 50)
    X_train, y_train, X_val, y_val = create_dataset()
    
    # Bước 2: Load processed dataset (nếu đã có sẵn)
    print("\n" + "=" * 50)
    print("BƯỚC 2: Load processed dataset")
    print("=" * 50)
    X_train, y_train, X_val, y_val = load_processed_dataset()