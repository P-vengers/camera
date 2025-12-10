import numpy as np
import cv2
import json

# ==========================================
# [입력 1] 로봇 좌표 (메모장에 적은 것 옮겨 적기)
# 순서가 카메라 데이터와 똑같아야 합니다! (단위: mm)
# ==========================================
ROBOT_COORDS = [
    [367.44, 6.02, 20],
    [515.38, -97.56, 20],
    [265.58, -99.16,20],
    [513.51, 151.65, 20],
    [265.29, 150.85, 20],
    [437.62, 40.23, 20],
    [316.73, 101.35, 20],
    [348.14, -51.37, 20],
    [386.6, 113.24, 20],
    [480.48, -72.43, 20],
    [451.19, -13.64, 106.94],
    [391.46, -19.09, 125.47],
    [333.27, 35.7, 110.15],
    [343.33, 60.55, 111.96],
    [340.25, 91.11, 100.14],
    [382.7, 114.74, 115.09],
    [419.15, 72.89, 108.99],
    [423.49, 49.69, 108.93]
]

# ==========================================
# [입력 2] 카메라 좌표 (저장된 파일에서 자동 로드)
# ==========================================
jsonl_file = "calibration_data_final.jsonl"

cam_coords_list = []

print(f"[정보] {jsonl_file} 로드 중...")
try:
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 미터(m) 단위 데이터를 밀리미터(mm)로 변환
            x_mm = data["X_m"] * 1000.0
            y_mm = data["Y_m"] * 1000.0
            z_mm = data["Z_m"] * 1000.0
            cam_coords_list.append([x_mm, y_mm, z_mm])
            
    print(f"[성공] 카메라 데이터 {len(cam_coords_list)}개 로드 완료.")

except FileNotFoundError:
    print("[에러] 데이터 파일이 없습니다. 1단계(데이터 수집)부터 진행하세요.")
    exit()

# 데이터 개수 확인
if len(ROBOT_COORDS) != len(cam_coords_list):
    print(f"[경고] 데이터 개수가 맞지 않습니다!")
    print(f" - 로봇 좌표: {len(ROBOT_COORDS)}개")
    print(f" - 카메라 좌표: {len(cam_coords_list)}개")
    print(" -> 개수를 맞춰주세요.")
    exit()

# Numpy 배열 변환
robot_points = np.array(ROBOT_COORDS, dtype=np.float32)
cam_points = np.array(cam_coords_list, dtype=np.float32)

# ==========================================
# [핵심] 캘리브레이션 (estimateAffine3D)
# ==========================================
print("\n--- 캘리브레이션 계산 중... ---")

# 이상치(Outlier)를 제거하며 최적의 행렬 계산
retval, T_affine, inliers = cv2.estimateAffine3D(cam_points, robot_points)

if retval:
    print("\n✅ 캘리브레이션 성공!")
    print("="*50)
    print("TRANSFORMATION_MATRIX = np.array([")
    for row in T_affine:
        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}],")
    print("])")
    print("="*50)
    
    # 검증 (오차 확인)
    total_error = 0
    print("\n[검증 결과]")
    for i in range(len(cam_points)):
        pt_cam = np.append(cam_points[i], 1.0) # [x, y, z, 1]
        pt_pred = np.dot(T_affine, pt_cam)     # 예측된 로봇 좌표
        
        pt_real = robot_points[i]
        error = np.linalg.norm(pt_pred - pt_real) # 거리 차이
        
        total_error += error
        print(f"#{i+1:02d} 오차: {error:.2f} mm")
        
    avg_error = total_error / len(cam_points)
    print(f"\n평균 오차: {avg_error:.2f} mm")
    
    if avg_error < 5.0:
        print("🎉 아주 훌륭합니다! 주사 프로젝트 진행 가능!")
    else:
        print("⚠️ 오차가 좀 큽니다. 데이터 수집을 더 신중하게 다시 해보세요.")

else:
    print("❌ 계산 실패. 데이터가 너무 적거나(최소 4개 필요) 일직선상에 있습니다.")
