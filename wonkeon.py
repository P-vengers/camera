import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ================== 설정값 ==================
NUM_LM = 478                 # FaceMesh 정제 모드 랜드마크 개수
MIN_DEPTH_M = 0.15           # 사용 깊이 범위 (m)
MAX_DEPTH_M = 1.0
MIN_SAMPLES_PER_POINT = 5    # 랜드마크별 최소 샘플 프레임 수
Z_VIS_SCALE = 2.0            # 시각화에서 깊이 과장 배수


# ================== (1) 단일 View 스캔 ==================
def scan_one_view(view_idx: int):
    """
    한 뷰(view)에서 얼굴을 고정한 상태로 's' 를 눌러 여러 프레임을 스캔.
    각 랜드마크에 대해 (X,Y,Z)을 여러 개 모아서
    -> 깊이(Z) 중앙값 기준으로 이상치 제거 -> 평균 X,Y,Z 반환.
    단위는 "미터".
    """
    print(f"\n[뷰 {view_idx}] RealSense 초기화...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("[ERROR] RealSense 시작 실패:", e)
        return None, None, True   # quit_flag=True

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # 랜드마크별로 모든 (X,Y,Z) 샘플을 저장
    all_samples = [[] for _ in range(NUM_LM)]

    scanning = False
    has_data = False
    quit_flag = False

    print(f"[뷰 {view_idx}] 준비 완료.")
    print("   - 's' : 스캔 시작 / 종료 (스캔 중에는 얼굴/카메라 절대 움직이지 말기)")
    print("   - 'q' 또는 ESC : 전체 종료")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            h, w, _ = color_img.shape

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            disp = color_img.copy()

            if results.multi_face_landmarks:
                lmks = results.multi_face_landmarks[0].landmark

                # 2D 랜드마크 표시
                for lm in lmks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(disp, (px, py), 1, (0, 255, 0), -1)

                if scanning:
                    intr = color_frame.profile.as_video_stream_profile().intrinsics

                    for idx, lm in enumerate(lmks):
                        if idx >= NUM_LM:
                            break

                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        if not (0 <= px < w and 0 <= py < h):
                            continue

                        d = depth_img[py, px]
                        if d == 0:
                            continue

                        d_m = d * depth_scale
                        if not (MIN_DEPTH_M <= d_m <= MAX_DEPTH_M):
                            continue

                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [px, py], d_m)
                        all_samples[idx].append([X, Y, Z])
                        has_data = True

                    cv2.putText(
                        disp,
                        f"[뷰 {view_idx}] SCANNING... (s=stop)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        disp,
                        f"[뷰 {view_idx}] Press 's' to start scan",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
            else:
                cv2.putText(
                    disp,
                    "Face NOT detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(f"View {view_idx}", disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[사용자 종료 요청]")
                quit_flag = True
                break

            if key == ord("s"):
                scanning = not scanning
                if scanning:
                    print(f"[뷰 {view_idx}] 스캔 시작")
                else:
                    print(f"[뷰 {view_idx}] 스캔 종료")
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if quit_flag:
        return None, None, True

    if not has_data:
        print("[뷰] 유효 스캔 데이터 없음")
        return None, None, False

    # ---------- median 기반 outlier 제거 + 평균 ----------
    mean_pts = np.zeros((NUM_LM, 3), dtype=np.float64)
    counts = np.zeros((NUM_LM,), dtype=np.int32)

    for idx, samples in enumerate(all_samples):
        if len(samples) < MIN_SAMPLES_PER_POINT:
            continue

        arr = np.asarray(samples)  # (N, 3) in meters
        z = arr[:, 2]
        z_med = np.median(z)

        # 깊이 2cm 이상 튀는 값 제거
        good = np.abs(z - z_med) < 0.02
        arr_good = arr[good]
        if arr_good.size == 0:
            continue

        mean_pts[idx] = arr_good.mean(axis=0)
        counts[idx] = arr_good.shape[0]

    valid_mask = counts > 0
    if not np.any(valid_mask):
        print("[뷰] 유효 포인트가 부족합니다.")
        return None, None, False

    print(f"[뷰 {view_idx}] 유효 랜드마크 수: {int(valid_mask.sum())}")
    print(f"[뷰 {view_idx}] 랜드마크당 평균 샘플 수: {counts[valid_mask].mean():.1f}")

    # 내부 단위는 계속 "미터"로 유지
    return mean_pts, valid_mask, False


# ================== (2) 여러 뷰 스캔 ==================
def capture_multiview():
    views = []
    view_idx = 0

    print("=======================================")
    print("   멀티뷰 얼굴 스캔")
    print("   1) 정면에서 's' 로 2~3초 스캔")
    print("   2) 얼굴이나 카메라 각도 바꾸고 또 's'")
    print("   3) 다 찍었으면 스캔 중이 아닐 때 'q'/ESC")
    print("=======================================\n")

    while True:
        print(f"[INFO] 새로운 뷰 스캔 시작 (현재 {len(views)}개 저장됨)")
        pts, mask, quit_flag = scan_one_view(view_idx)

        if quit_flag:
            break

        if pts is None or mask is None:
            print(f"[WARN] 뷰 {view_idx} 데이터 없음")
        else:
            print(f"[뷰 {view_idx}] 유효 랜드마크 수: {int(mask.sum())}")
            views.append((pts, mask))
            view_idx += 1

    return views


# ================== (3) 두 뷰 정렬 (Kabsch) ==================
def align_view_to_reference(ref_pts, ref_mask, view_pts, view_mask):
    """
    ref_pts, view_pts : 모두 (478, 3) in meters
    """
    common = ref_mask & view_mask
    idxs = np.where(common)[0]

    if len(idxs) < 15:
        print("[WARN] 공통 포인트가 너무 적어 정렬 스킵")
        return view_pts, view_mask

    P = view_pts[idxs]  # moving
    Q = ref_pts[idxs]   # reference

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = Q.mean(axis=0) - R @ P.mean(axis=0)

    aligned = (R @ view_pts.T).T + t
    print(f"[정렬] 공통 포인트 {len(idxs)}개 사용")

    return aligned, view_mask


# ================== (4) 여러 뷰 Fuse ==================
def fuse_views(views):
    if not views:
        return None, None

    base_pts, base_mask = views[0]
    fused_pts = base_pts.copy()
    fused_counts = base_mask.astype(np.int32)

    for i in range(1, len(views)):
        pts, mask = views[i]
        print(f"[Fuse] 뷰 {i}를 기준에 정렬 중...")

        aligned, new_mask = align_view_to_reference(
            fused_pts,
            fused_counts > 0,
            pts,
            mask,
        )

        for idx in range(NUM_LM):
            if not new_mask[idx]:
                continue

            if fused_counts[idx] == 0:
                fused_pts[idx] = aligned[idx]
                fused_counts[idx] = 1
            else:
                c = fused_counts[idx]
                fused_pts[idx] = (fused_pts[idx] * c + aligned[idx]) / (c + 1)
                fused_counts[idx] += 1

    final_mask = fused_counts > 0
    print(f"[Fuse 완료] 최종 유효 포인트: {int(final_mask.sum())}개")

    return fused_pts, final_mask


# ================== (5) Matplotlib 정면/측면 시각화 ==================
def visualize_single_model(points_m, valid_mask):
    """
    points_m : (478,3) in meters
    """
    pts = points_m[valid_mask]
    # mm 단위로 변환 + 깊이 과장
    xs = pts[:, 0] * 1000.0
    ys = pts[:, 1] * 1000.0
    zs = pts[:, 2] * 1000.0 * Z_VIS_SCALE

    fig = plt.figure(figsize=(10, 5))

    # 정면
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(xs, ys, zs, s=4)
    ax1.set_title("Front View")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel(f"Z (mm) x{Z_VIS_SCALE}")
    ax1.view_init(elev=20, azim=-90)

    # 측면
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(xs, ys, zs, s=4)
    ax2.set_title("Side View")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel(f"Z (mm) x{Z_VIS_SCALE}")
    # 오른쪽 옆에서 보는 방향
    ax2.view_init(elev=10, azim=0)

    plt.tight_layout()
    plt.show()


# ================== (6) main() ==================
def main():
    # 1) 멀티뷰 스캔
    views = capture_multiview()
    if not views:
        print("[ERROR] 스캔된 뷰가 없습니다.")
        return

    # 2) 정렬 + Fuse
    fused_pts, fused_mask = fuse_views(views)
    if fused_pts is None:
        print("[ERROR] Fuse 실패")
        return

    print("[INFO] 얼굴 모델 정렬/통합 완료")

    # 3) 정면/측면 시각화
    visualize_single_model(fused_pts, fused_mask)


# ================== 실행 ==================
if __name__ == "__main__":
    main()
