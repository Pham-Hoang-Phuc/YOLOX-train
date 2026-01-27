import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 1. Cấu hình số lượng class
        self.num_classes = 72

        # 2. Đường dẫn dữ liệu
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --- 3. TỐI ƯU CHO 10 EPOCH ---
        
        # Tổng số epoch
        self.max_epoch = 10
        
        # Warmup: Mặc định là 5, nhưng với 10 epoch thì chỉ nên để 1 
        # để model sớm bước vào giai đoạn học chính thức.
        self.warmup_epochs = 1
        
        # No-aug epochs: Đây là giai đoạn tắt các phép biến đổi ảnh (augmentation) mạnh ở cuối.
        # Với 10 epoch, hãy dành 2 epoch cuối để model ổn định lại.
        self.no_aug_epochs = 2
        
        # Khoảng cách đánh giá (Evaluation): Nên để là 1 để mỗi epoch đều biết model tiến triển ra sao.
        self.eval_interval = 1 
        
        # Khoảng cách in log: In log mỗi 10 lần lặp (iteration) để bạn theo dõi sát sao hơn.
        self.print_interval = 10

        # Tốc độ học tối thiểu: Giữ nguyên hoặc giảm nhẹ để learning rate không bị tụt quá nhanh.
        self.min_lr_ratio = 0.05
        
        # Cấu hình worker tùy theo cấu hình máy ảo (thường 4 là ổn)
        self.data_num_workers = 4