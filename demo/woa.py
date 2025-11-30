import numpy as np
import math
import random
import matplotlib.pyplot as plt

# 1. Định nghĩa hàm mục tiêu (Objective Function)
def objective_function(x):
    """
    Hàm Sphere: f(x) = x1^2 + x2^2 + ... + xn^2
    Giá trị tối ưu là 0 tại vị trí [0, 0, ..., 0]
    """
    return np.sum(x**2)

# 2. Thuật toán WOA
def WOA(search_agents_no, max_iter, lb, ub, dim):
    """
    search_agents_no: Số lượng cá voi (quần thể)
    max_iter: Số vòng lặp tối đa
    lb: Giới hạn dưới (Lower Bound)
    ub: Giới hạn trên (Upper Bound)
    dim: Số chiều không gian
    """
    
    # Khởi tạo quần thể cá voi ngẫu nhiên
    positions = np.random.uniform(lb, ub, (search_agents_no, dim))
    
    # Khởi tạo vị trí và điểm số của con đầu đàn (Con mồi)
    leader_pos = np.zeros(dim)
    leader_score = float("inf") # Khởi tạo là vô cùng lớn (để tìm min)
    
    # Lưu lịch sử để vẽ biểu đồ
    convergence_curve = []
    
    # Bắt đầu vòng lặp thời gian
    t = 0
    while t < max_iter:
        
        # Kiểm tra biên và tính toán độ thích nghi (Fitness)
        for i in range(search_agents_no):
            # Kéo cá voi về biên nếu nó bơi ra ngoài
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            # Tính fitness
            fitness = objective_function(positions[i, :])
            
            # Cập nhật con đầu đàn nếu tìm thấy vị trí tốt hơn
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()
        
        # Cập nhật tham số a (giảm tuyến tính từ 2 xuống 0)
        a = 2 - 2 * t / max_iter
        
        # Cập nhật vị trí từng con cá voi
        for i in range(search_agents_no):
            r1 = random.random() # Số ngẫu nhiên [0,1]
            r2 = random.random() # Số ngẫu nhiên [0,1]
            
            A = 2 * a * r1 - a  # Vector A
            C = 2 * r2          # Vector C
            
            # Các tham số cho hình xoắn ốc
            b = 1
            l = random.uniform(-1, 1)
            p = random.random() # Xác suất chọn cơ chế
            
            # --- Logic di chuyển của WOA ---
            
            if p >= 0.5:
                # 1. Giai đoạn Tấn công bong bóng (Xoắn ốc)
                # D' = |Best - Current|
                dist_to_leader = np.abs(leader_pos - positions[i, :])
                # Công thức: X(t+1) = D' * e^bl * cos(2pi*l) + Best
                positions[i, :] = dist_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
                
            else: # p < 0.5
                if abs(A) < 1:
                    # 2. Giai đoạn Bao vây con mồi (Exploitation)
                    # D = |C * Best - Current|
                    D = np.abs(C * leader_pos - positions[i, :])
                    # X(t+1) = Best - A * D
                    positions[i, :] = leader_pos - A * D
                else:
                    # 3. Giai đoạn Tìm kiếm con mồi (Exploration / Search)
                    # Chọn ngẫu nhiên một con cá voi khác
                    random_whale_idx = random.randint(0, search_agents_no - 1)
                    random_whale_pos = positions[random_whale_idx, :]
                    
                    # D = |C * RandWhale - Current|
                    D = np.abs(C * random_whale_pos - positions[i, :])
                    # X(t+1) = RandWhale - A * D
                    positions[i, :] = random_whale_pos - A * D
        
        # Lưu lại kết quả tốt nhất vòng lặp này
        convergence_curve.append(leader_score)
        
        # In tiến độ
        if t % 10 == 0:
            print(f"Vòng lặp {t}: Fitness tốt nhất = {leader_score}")
            
        t += 1
        
    return leader_pos, leader_score, convergence_curve

# --- CHẠY THỬ NGHIỆM ---

# Tham số cài đặt
so_luong_ca_voi = 30
so_vong_lap = 100
gioi_han_duoi = -100
gioi_han_tren = 100
so_chieu = 30 # Bài toán 30 chiều

print("=== BẮT ĐẦU SĂN MỒI ===")
best_pos, best_score, curve = WOA(so_luong_ca_voi, so_vong_lap, gioi_han_duoi, gioi_han_tren, so_chieu)

print("\n=== KẾT QUẢ ===")
print(f"Vị trí tối ưu tìm được: {best_pos}")
print(f"Giá trị mục tiêu tốt nhất (gần 0 là tốt): {best_score}")

# Vẽ biểu đồ hội tụ
plt.figure(figsize=(10, 5))
plt.plot(curve, color='blue', linewidth=2)
plt.title('Biểu đồ hội tụ của thuật toán WOA (Convergence Curve)')
plt.xlabel('Vòng lặp (Iteration)')
plt.ylabel('Giá trị hàm mục tiêu (Fitness)')
plt.yscale('log') # Dùng thang log để nhìn rõ sự giảm xuống gần 0
plt.grid(True)
plt.show()