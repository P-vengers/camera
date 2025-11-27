import cv2
import numpy as np
import mediapipe as mp
import time
import json
import math
import pyrealsense2 as rs  # Intel RealSense SDK

# =======================
# 하이레벨 파라미터
# =======================
SCAN_DURATION = 5.0   # 스캔 시간(초) ≈ 4~5초
SAFE_RADIUS   = 50.0  # 스캔 후 얼굴 중심이 이 픽셀 거리 이상 벗어나면 경고
GRID_STEP     = 8     # face_mask 위 격자 간격 (픽셀)
NEIGHBOR_R_M  = 0.005 # 법선추정용 이웃반경 (미터) ≈ 5mm

# =======================
# MediaPipe FaceMesh 인덱스들
# =======================

# 얼굴 외곽(얼굴 타원)로 자주 쓰이는 FaceMesh 인덱스들 (대략적인 oval)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# 왼/오 볼 윤곽을 이루는 랜드마크 인덱스 (시각화용)
LEFT_CHEEK_IDX  = [234,  93, 132,  58, 172, 136, 150, 176, 148, 152]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 400, 377, 152]

# 눈/코/입 근처를 대략적으로 정의하기 위한 랜드마크 몇 개
LEFT_EYE_CENTER_IDX  = [33, 133]   # 왼쪽 눈 근처 두 점 평균
RIGHT_EYE_CENTER_IDX = [362, 263]  # 오른쪽 눈 근처 두 점 평균
NOSE_CENTER_IDX      = [1, 4]      # 코 중앙/끝 근처
MOUTH_CENTER_IDX     = [13, 14]    # 입 윗/아랫 부분 근처

# 눈/코/입 주변을 제외하기 위한 픽셀 반경 (대략)
EXCLUDE_EYE_RADIUS   = 35
EXCLUDE_NOSE_RADIUS  = 30
EXCLUDE_MOUTH_RADIUS = 40


# =======================
# 유틸 함수들
# =======================

def get_point_mean(landmarks, idx_list, w, h):
    """landmarks에서 idx_list에 해당하는 좌표들 평균 (u, v) 리턴"""
    pts = []
    for idx in idx_list:
        lm = landmarks.landmark[idx]
        u = int(lm.x * w)
        v = int(lm.y * h)
        pts.append((u, v))
    if not pts:
        return None
    mx = int(sum(p[0] for p in pts) / len(pts))
    my = int(sum(p[1] for p in pts) / len(pts))
    return mx, my


def estimate_normals(points_xyz, neighbor_radius=0.005):
    """
    points_xyz: 리스트 [ { "X_m": X, "Y_m": Y, "Z_m": Z, ... }, ... ]
    neighbor_radius: 이웃 반경 (meter 기준, 예: 0.005 = 5mm)

    반환:
        normals: 리스트 [ (nx, ny, nz), ... ]  (points_xyz와 같은 순서)
    """
    if not points_xyz:
        return []

    # (N, 3) numpy array로 변환
    P = np.array([[p["X_m"], p["Y_m"], p["Z_m"]] for p in points_xyz], dtype=np.float32)
    N = P.shape[0]
    normals = []

    for i in range(N):
        p = P[i]

        # 모든 점과 거리 계산 (단순 O(N^2) - 포인트 수가 많으면 나중에 최적화 가능)
        diff = P - p  # (N, 3)
        dist = np.linalg.norm(diff, axis=1)

        # 자기 자신 제외 + neighbor_radius 이하인 점들만 이웃으로 사용
        mask = (dist > 0) & (dist <= neighbor_radius)
        neighbors = P[mask]

        if neighbors.shape[0] < 3:
            # 이웃이 너무 적으면 기본값 (0, 0, 1) 사용
            normals.append((0.0, 0.0, 1.0))
            continue

        # 이웃 점들의 평균을 기준으로 중심화
        mu = neighbors.mean(axis=0)
        Q = neighbors - mu  # (M, 3)

        # 공분산 행렬
        C = Q.T @ Q  # (3, 3)

        # 고유값/고유벡터 계산
        vals, vecs = np.linalg.eigh(C)
        # 가장 작은 고유값에 해당하는 고유벡터가 평면의 법선
        normal = vecs[:, np.argmin(vals)]

        # 방향(부호)은 일단 정규화만 하고 그대로 사용
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        normals.append((float(normal[0]), float(normal[1]), float(normal[2])))

    return normals


# =======================
# 메인 파이프라인
# =======================

def main():
    # ==============
    # 1. RealSense 초기화
    # ==============
    pipeline = rs.pipeline()
    config = rs.config()

    # 컬러 + 뎁스 스트림 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # depth를 color에 align
    align_to = rs.stream.color
    align = rs.align(align_to)

    # depth 카메라 intrinsics (X,Y,Z 계산용)
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    print(f"[INFO] RealSense depth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ==============
    # 2. MediaPipe FaceMesh 초기화
    # ==============
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ==============
    # 스캔/안전영역 관리용 변수
    # ==============
    scan_start_time = time.time()
    scanning = True  # True: 얼굴 영역 좌표 스캔 모드, False: 안전영역 모니터링 모드

    safe_center = None  # 스캔 종료 시 얼굴 기준 중심
    was_inside_safe = True  # 바로 직전 프레임에서 안전영역 안이었는지

    last_scan_points = []  # 스캔 마지막 프레임에서의 후보 포인트들 (X,Y,Z 포함)

    try:
        while True:
            now = time.time()
            elapsed = now - scan_start_time

            # 스캔 시간 종료 체크
            if scanning and elapsed >= SCAN_DURATION:
                scanning = False
                print(f"[INFO] 스캔 종료 ({elapsed:.2f}초). 이제 안전 영역만 모니터링합니다.")
                # safe_center는 아래 FaceMesh 처리에서 최신 얼굴 중심으로 설정될 것

            # ==============
            # 3. D435에서 프레임 받기 (color + depth aligned)
            # ==============
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # BGR
            h, w, _ = frame.shape

            # ==============
            # 4. FaceMesh로 얼굴 polygon + 볼/얼굴 중심 계산
            # ==============
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            left_poly = []
            right_poly = []
            face_oval_poly = []
            face_center = None  # 현재 프레임 기준 얼굴 중심 (볼 기준)

            # 눈/코/입 중심
            left_eye_center = None
            right_eye_center = None
            nose_center = None
            mouth_center = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # --- 얼굴 외곽(oval) polygon ---
                for idx in FACE_OVAL_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    face_oval_poly.append((u, v))
                if len(face_oval_poly) >= 3:
                    oval_np = np.array(face_oval_poly, np.int32)
                    cv2.polylines(frame, [oval_np], True, (0, 255, 0), 1)

                # --- 왼쪽 볼 (시각화/center 계산용) ---
                for idx in LEFT_CHEEK_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    left_poly.append((u, v))
                    cv2.circle(frame, (u, v), 2, (0, 0, 255), -1)  # 빨강 점

                # --- 오른쪽 볼 (시각화/center 계산용) ---
                for idx in RIGHT_CHEEK_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    right_poly.append((u, v))
                    cv2.circle(frame, (u, v), 2, (255, 0, 0), -1)  # 파랑 점

                # 얼굴 중심(볼들 평균 위치) 계산
                cheek_points = left_poly + right_poly
                if cheek_points:
                    cx_mean = int(sum(p[0] for p in cheek_points) / len(cheek_points))
                    cy_mean = int(sum(p[1] for p in cheek_points) / len(cheek_points))
                    face_center = (cx_mean, cy_mean)
                    cv2.circle(frame, face_center, 4, (0, 255, 255), -1)  # 노랑

                    # 스캔이 끝난 뒤, 첫 얼굴 중심을 안전 기준으로 설정
                    if not scanning and safe_center is None:
                        safe_center = face_center
                        print(f"[INFO] 안전 영역 기준 중심 설정: {safe_center}")

                # 눈/코/입 중심 대략 계산
                left_eye_center  = get_point_mean(face_landmarks, LEFT_EYE_CENTER_IDX,  w, h)
                right_eye_center = get_point_mean(face_landmarks, RIGHT_EYE_CENTER_IDX, w, h)
                nose_center      = get_point_mean(face_landmarks, NOSE_CENTER_IDX,      w, h)
                mouth_center     = get_point_mean(face_landmarks, MOUTH_CENTER_IDX,     w, h)

                # 시각화
                for c, col in [
                    (left_eye_center,  (0, 255, 255)),
                    (right_eye_center, (0, 255, 255)),
                    (nose_center,      (0, 165, 255)),
                    (mouth_center,     (255, 255, 0)),
                ]:
                    if c is not None:
                        cv2.circle(frame, c, 3, col, -1)

            # ==============
            # 5. 스캔 모드일 때: face_mask 위에 GRID를 씌워서 후보 포인트(X,Y,Z) 수집
            #    (마지막 스캔 프레임만 저장)
            # ==============
            if scanning and face_oval_poly:
                face_mask = np.zeros((h, w), dtype=np.uint8)

                # 1) 얼굴 외곽(oval) 채우기
                cv2.fillPoly(face_mask, [np.array(face_oval_poly, np.int32)], 255)

                # 2) 눈/코/입 주변 영역을 0으로 만들어서 제외
                def erase_circle(center, radius):
                    if center is None:
                        return
                    cx_i, cy_i = center
                    cv2.circle(face_mask, (cx_i, cy_i), radius, 0, -1)

                erase_circle(left_eye_center,  EXCLUDE_EYE_RADIUS)
                erase_circle(right_eye_center, EXCLUDE_EYE_RADIUS)
                erase_circle(nose_center,      EXCLUDE_NOSE_RADIUS)
                erase_circle(mouth_center,     EXCLUDE_MOUTH_RADIUS)

                # 3) 얼굴 바운딩 박스 안에서 GRID 순회
                xs_all = [p[0] for p in face_oval_poly]
                ys_all = [p[1] for p in face_oval_poly]
                x_min = max(min(xs_all), 0)
                x_max = min(max(xs_all), w - 1)
                y_min = max(min(ys_all), 0)
                y_max = min(max(ys_all), h - 1)

                points_this_frame = []

                for v in range(y_min, y_max + 1, GRID_STEP):
                    for u in range(x_min, x_max + 1, GRID_STEP):
                        if face_mask[v, u] != 255:
                            continue

                        depth_m = float(depth_frame.get_distance(int(u), int(v)))
                        if depth_m <= 0:
                            continue

                        # 카메라 좌표계 기준 3D (X, Y, Z)
                        X = (u - cx) / fx * depth_m
                        Y = (v - cy) / fy * depth_m
                        Z = depth_m

                        points_this_frame.append({
                            "u": int(u),
                            "v": int(v),
                            "depth_m": depth_m,
                            "X_m": X,
                            "Y_m": Y,
                            "Z_m": Z,
                        })

                        # 시각화: GRID 포인트
                        cv2.circle(frame, (int(u), int(v)), 1, (0, 255, 0), -1)

                # 마지막 스캔 프레임의 포인트들을 계속 갱신
                last_scan_points = points_this_frame

            # ==============
            # 6. 스캔 이후: 얼굴 안전 영역 모니터링
            # ==============
            if not scanning and safe_center is not None and face_center is not None:
                dx = face_center[0] - safe_center[0]
                dy = face_center[1] - safe_center[1]
                dist = math.hypot(dx, dy)

                # 안전 영역 시각화 (원)
                cv2.circle(frame, safe_center, int(SAFE_RADIUS), (0, 255, 0), 1)

                now_inside_safe = dist <= SAFE_RADIUS

                # 안에 있다가 처음 밖으로 나가는 순간에만 경고 출력
                if was_inside_safe and not now_inside_safe:
                    print("얼굴이 안전 영역을 벗어났습니다. 정지합니다.")

                was_inside_safe = now_inside_safe

            # ==============
            # 7. 화면 표시
            # ==============
            if scanning:
                cv2.putText(
                    frame,
                    f"SCANNING (face skin grid)... {elapsed:.1f}s / {SCAN_DURATION:.1f}s",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "SAFE MONITORING",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("D435 + Face Skin GRID + Safe Zone", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        # ==============
        # 8. 스캔 결과 후처리: 마지막 스캔 프레임 포인트들로 법선 추정
        # ==============
        try:
            if last_scan_points:
                print(f"[INFO] 스캔 포인트 개수: {len(last_scan_points)}")
                normals = estimate_normals(last_scan_points, neighbor_radius=NEIGHBOR_R_M)
                print(f"[INFO] 법선 추정 완료. (points={len(normals)})")

                # 포인트+법선 합치기
                output = []
                for p, n in zip(last_scan_points, normals):
                    out = {
                        "u": p["u"],
                        "v": p["v"],
                        "depth_m": p["depth_m"],
                        "X_m": p["X_m"],
                        "Y_m": p["Y_m"],
                        "Z_m": p["Z_m"],
                        "nx": n[0],
                        "ny": n[1],
                        "nz": n[2],
                    }
                    output.append(out)

                with open("scan_points_with_normals.json", "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

                print("[INFO] scan_points_with_normals.json 파일로 포인트+법선을 저장했습니다.")
            else:
                print("[WARN] last_scan_points가 비어 있습니다. 스캔에 실패했을 수 있습니다.")

        except Exception as e:
            print(f"[WARN] 스캔 결과/법선 저장 중 오류: {e}")

        face_mesh.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
