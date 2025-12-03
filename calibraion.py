import numpy as np
import json

def compute_rigid_transform(points_cam, points_robot):
    """
    points_cam: Nx3 (카메라 좌표계, mm 단위)
    points_robot: Nx3 (로봇 좌표계, mm 단위)
    """
    A = np.asarray(points_cam, dtype=np.float64)
    B = np.asarray(points_robot, dtype=np.float64)
    assert A.shape == B.shape
    N = A.shape[0]
    assert N >= 3, f"최소 3점 이상 필요합니다. 현재 N={N}"

    # 중심 정렬
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB

    # SVD
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection 보정
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cB - R @ cA
    return R, t

def main():
    # ==========================
    # 1) 여기에 네가 측정한 값들 넣으면 됨
    # ==========================
    # cam_points_m: 카메라 좌표 (X_m, Y_m, Z_m), 단위: meter
    cam_points_m = [
        # 예시:
        # [0.0599, 0.0275, 0.3980],
        # [X_m2,   Y_m2,   Z_m2  ],
        # ...
    ]

    # robot_points_mm: 로봇 좌표 (Xr, Yr, Zr), 단위: mm
    robot_points_mm = [
        # [367.44, 6.02, 20.0],
        # [Xr2,    Yr2,  Zr2 ],
        # ...
    ]

    cam_points_m = np.asarray(cam_points_m, dtype=np.float64)
    robot_points_mm = np.asarray(robot_points_mm, dtype=np.float64)

    assert cam_points_m.shape == robot_points_mm.shape
    assert cam_points_m.shape[0] >= 3

    # 카메라 좌표를 mm로 맞춰줌
    cam_points_mm = cam_points_m * 1000.0

    # ==========================
    # 2) R, t 계산
    # ==========================
    R, t = compute_rigid_transform(cam_points_mm, robot_points_mm)

    print("=== Rotation R (cam -> robot) ===")
    print(R)
    print("\n=== Translation t (mm) ===")
    print(t)

    # ==========================
    # 3) 재투영 오차 확인
    # ==========================
    preds = (R @ cam_points_mm.T).T + t  # Nx3
    errors = np.linalg.norm(preds - robot_points_mm, axis=1)

    print("\n=== Per-point error (mm) ===")
    for i, e in enumerate(errors):
        print(f"Point {i+1}: {e:.3f} mm")

    print(f"\nRMS error: {np.sqrt((errors**2).mean()):.3f} mm")

    # ==========================
    # 4) JSON으로 저장
    # ==========================
    extrinsic = {
        "R": R.tolist(),
        "t_mm": t.tolist()
    }
    with open("cam_to_robot_manual.json", "w", encoding="utf-8") as f:
        json.dump(extrinsic, f, indent=2)
    print("\n[INFO] cam_to_robot_manual.json 저장 완료")


if __name__ == "__main__":
    main()
