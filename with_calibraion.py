import cv2
import numpy as np
import mediapipe as mp
import time
import json
import math
import socket
import pyrealsense2 as rs
import random
import os

# =======================
# 로봇 인터페이스
# =======================
class RobotInterface:
    """
    Doosan DRL ↔ Python 소켓 서버와 연결되어 있다고 가정.
    - DRL 쪽 프로토콜:
        1) 파이썬이 {"cmd": "get_pose"} + '\n' 전송
        2) DRL이 {"x": mm, "y": mm, "z": mm} + '\n' 으로 응답
    """

    def __init__(self, host="127.0.0.1", port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        # 필요하면 self.sock.settimeout(2.0) 같은 타임아웃도 설정 가능

    def _recv_line(self):
        """
        '\n' 이 나올 때까지 수신 후, 한 줄(str)로 반환
        """
        buf = b""
        while True:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("로봇 서버와의 연결이 끊어졌습니다.")
            buf += chunk
            if b"\n" in buf:
                line, _, _ = buf.partition(b"\n")
                return line.decode("utf-8")

    def get_tool_position(self):
        """
        현재 TCP 위치를 [x, y, z] (meter, base 기준)로 반환.
        """
        req = {"cmd": "get_pose"}
        msg = json.dumps(req) + "\n"
        self.sock.sendall(msg.encode("utf-8"))

        line = self._recv_line()
        data = json.loads(line)

        # DRL 쪽에서 mm 단위로 보낸다고 가정
        x_m = data["x"] / 1000.0
        y_m = data["y"] / 1000.0
        z_m = data["z"] / 1000.0

        return np.array([x_m, y_m, z_m], dtype=np.float32)

    def close(self):
        try:
            self.sock.close()
        except:
            pass

# =======================
# 공통 유틸
# =======================

SCAN_DURATION = 5.0
SAFE_RADIUS   = 50.0
DESIRED_SPACING_MM = 10.0
NEIGHBOR_R_M  = 0.015

# FaceMesh 관련 인덱스 (네 코드 그대로)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]
LEFT_CHEEK_IDX  = [234,  93, 132,  58, 172, 136, 150, 176, 148, 152]
RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 400, 377, 152]
LEFT_EYE_CENTER_IDX  = [33, 133]
RIGHT_EYE_CENTER_IDX = [362, 263]
NOSE_CENTER_IDX      = [1, 4]
MOUTH_CENTER_IDX     = [13, 14]

EXCLUDE_EYE_RADIUS   = 35
EXCLUDE_NOSE_RADIUS  = 30
EXCLUDE_MOUTH_RADIUS = 40


def get_point_mean(landmarks, idx_list, w, h):
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


def estimate_normals(points_xyz, neighbor_radius=0.01):
    if not points_xyz:
        return []
    P = np.array([[p["X_m"], p["Y_m"], p["Z_m"]] for p in points_xyz], dtype=np.float32)
    N = P.shape[0]
    normals = []
    for i in range(N):
        p = P[i]
        diff = P - p
        dist = np.linalg.norm(diff, axis=1)
        mask = (dist > 0) & (dist <= neighbor_radius)
        neighbors = P[mask]
        if neighbors.shape[0] < 3:
            normals.append((0.0, 0.0, 1.0))
            continue
        mu = neighbors.mean(axis=0)
        Q = neighbors - mu
        C = Q.T @ Q
        vals, vecs = np.linalg.eigh(C)
        normal = vecs[:, np.argmin(vals)]
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        normals.append((float(normal[0]), float(normal[1]), float(normal[2])))
    return normals


def transform_cam_to_robot(point_cam, R, t):
    Xc = np.array(point_cam, dtype=np.float32).reshape(3, 1)
    Xr = R @ Xc + t.reshape(3, 1)
    return float(Xr[0, 0]), float(Xr[1, 0]), float(Xr[2, 0])


def transform_normal_cam_to_robot(normal_cam, R):
    n = np.array(normal_cam, dtype=np.float32).reshape(3, 1)
    nr = R @ n
    nr = nr / (np.linalg.norm(nr) + 1e-8)
    return float(nr[0, 0]), float(nr[1, 0]), float(nr[2, 0])


def normal_to_rpy(nx, ny, nz):
    import math
    z_axis = np.array([nx, ny, nz], dtype=np.float32)
    z_axis /= (np.linalg.norm(z_axis) + 1e-8)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(z_axis, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    x_axis = np.cross(up, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    ry = math.asin(-R[2, 0])
    cy = math.cos(ry)
    if abs(cy) < 1e-6:
        rx = 0.0
        rz = math.atan2(-R[0, 1], R[1, 1])
    else:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])

    return float(rx), float(ry), float(rz)


# =======================
# 0. RealSense 초기화
# =======================

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    print(f"[INFO] RealSense depth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    return pipeline, align, intr


# =======================
# 1~3. Rigid Transform Calibration
# =======================

def compute_rigid_transform(points_cam, points_robot):
    """
    points_cam: Nx3, 카메라 좌표계
    points_robot: Nx3, 로봇 좌표계 (base 기준)
    """
    A = np.asarray(points_cam, dtype=np.float32)
    B = np.asarray(points_robot, dtype=np.float32)
    assert A.shape == B.shape
    N = A.shape[0]
    assert N >= 3, "최소 3점 이상 필요"

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cB - R @ cA
    return R, t


def calibrate_rigid_transform(pipeline, align, intr, robot, num_points=6):
    """
    1. 초기 위치(카메라 고정)에서, 화면에 보이는 점들을 클릭해서 카메라 3D 좌표 수집
    2. 각 점마다 로봇 TCP를 그 점에 갖다 대고, 로봇 좌표 수집
    3. rigid transform R, t 계산
    """
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print("\n[CALIB] Rigid Transform Calibration 시작")
    print(f"[CALIB] 최소 {num_points}개 이상의 점을 수집합니다.")
    print("[CALIB] 절차:")
    print("  1) 프레임에서 원하는 점을 마우스로 클릭 → 카메라 3D 좌표 자동 계산")
    print("  2) 로봇 TCP를 그 실제 점에 맞추고, Enter를 누르면 로봇 좌표를 기록")
    print("  3) 이 과정을 num_points번 반복 후 R, t 계산\n")

    clicked = {"u": None, "v": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["u"] = x
            clicked["v"] = y
            print(f"[CALIB] 클릭: u={x}, v={y}")

    cv2.namedWindow("CalibrationView")
    cv2.setMouseCallback("CalibrationView", on_mouse)

    cam_points = []
    robot_points = []

    try:
        while len(cam_points) < num_points:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            disp = frame.copy()
            cv2.putText(disp, f"Click point #{len(cam_points)+1}/{num_points}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("CalibrationView", disp)
            key = cv2.waitKey(1) & 0xFF

            if clicked["u"] is not None:
                u = clicked["u"]
                v = clicked["v"]
                clicked["u"] = None
                clicked["v"] = None

                depth_m = float(depth_frame.get_distance(int(u), int(v)))
                if depth_m <= 0:
                    print("[CALIB] depth=0, 다른 점을 클릭하세요.")
                    continue

                X = (u - cx) / fx * depth_m
                Y = (v - cy) / fy * depth_m
                Z = depth_m
                p_cam = np.array([X, Y, Z], dtype=np.float32)
                print(f"[CALIB] 카메라 3D = {p_cam}")

                input("[CALIB] 이제 로봇 TCP를 해당 실제 점에 맞추고 Enter를 누르세요...")
                p_robot = robot.get_tool_position()  # [x,y,z] in meter
                print(f"[CALIB] 로봇 3D = {p_robot}")

                cam_points.append(p_cam)
                robot_points.append(p_robot)
                print(f"[CALIB] 현재 {len(cam_points)}/{num_points} 개 수집 완료\n")

            if key == ord('q'):
                print("[CALIB] 사용자 중단")
                break

    finally:
        cv2.destroyWindow("CalibrationView")

    if len(cam_points) < 3:
        raise RuntimeError("캘리브레이션 점이 부족합니다. (최소 3개)")

    R, t = compute_rigid_transform(cam_points, robot_points)
    print("[CALIB] R:\n", R)
    print("[CALIB] t:\n", t)

    extrinsic = {
        "R": R.tolist(),
        "t": t.tolist()
    }
    with open("cam_to_robot_rigid.json", "w") as f:
        json.dump(extrinsic, f, indent=2)
    print("[CALIB] cam_to_robot_rigid.json 저장 완료")

    return R, t


def load_rigid_transform():
    try:
        with open("cam_to_robot_rigid.json", "r") as f:
            data = json.load(f)
        R = np.array(data["R"], dtype=np.float32)
        t = np.array(data["t"], dtype=np.float32)
        return R, t
    except Exception as e:
        print(f"[WARN] rigid transform 로드 실패: {e}")
        return None, None


# =======================
# 4. 5초 얼굴 스캔 (기존 로직 거의 그대로)
# =======================

def scan_face_and_save():
    """
    5초간 얼굴 스캔 → skin grid + 법선 계산 → scan_points_with_normals.json 저장
    (네가 준 코드 거의 그대로, 5개 포인트 랜덤 선택은 사용 안 함)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    Z_REF = 0.35
    GRID_STEP = int(round((DESIRED_SPACING_MM / 1000.0) * fx / Z_REF))
    GRID_STEP = max(GRID_STEP, 1)

    print(f"[INFO] Target spacing = {DESIRED_SPACING_MM} mm @ Z≈{Z_REF} m → GRID_STEP = {GRID_STEP} px")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    scan_start_time = time.time()
    scanning = True
    safe_center = None
    was_inside_safe = True
    last_scan_points = []
    last_scan_frame = None

    try:
        while True:
            now = time.time()
            elapsed = now - scan_start_time
            if scanning and elapsed >= SCAN_DURATION:
                scanning = False
                print(f"[INFO] 스캔 종료 ({elapsed:.2f}초). 이제 안전 영역만 모니터링합니다.")

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h, w, _ = frame.shape

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            left_poly = []
            right_poly = []
            face_oval_poly = []
            face_center = None

            left_eye_center = None
            right_eye_center = None
            nose_center = None
            mouth_center = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                for idx in FACE_OVAL_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    face_oval_poly.append((u, v))
                if len(face_oval_poly) >= 3:
                    oval_np = np.array(face_oval_poly, np.int32)
                    cv2.polylines(frame, [oval_np], True, (0, 255, 0), 1)

                for idx in LEFT_CHEEK_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    left_poly.append((u, v))
                    cv2.circle(frame, (u, v), 2, (0, 0, 255), -1)

                for idx in RIGHT_CHEEK_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    right_poly.append((u, v))
                    cv2.circle(frame, (u, v), 2, (255, 0, 0), -1)

                cheek_points = left_poly + right_poly
                if cheek_points:
                    cx_mean = int(sum(p[0] for p in cheek_points) / len(cheek_points))
                    cy_mean = int(sum(p[1] for p in cheek_points) / len(cheek_points))
                    face_center = (cx_mean, cy_mean)
                    cv2.circle(frame, face_center, 4, (0, 255, 255), -1)
                    if not scanning and safe_center is None:
                        safe_center = face_center
                        print(f"[INFO] 안전 영역 기준 중심 설정: {safe_center}")

                left_eye_center  = get_point_mean(face_landmarks, LEFT_EYE_CENTER_IDX,  w, h)
                right_eye_center = get_point_mean(face_landmarks, RIGHT_EYE_CENTER_IDX, w, h)
                nose_center      = get_point_mean(face_landmarks, NOSE_CENTER_IDX,      w, h)
                mouth_center     = get_point_mean(face_landmarks, MOUTH_CENTER_IDX,     w, h)

                for c, col in [
                    (left_eye_center,  (0, 255, 255)),
                    (right_eye_center, (0, 255, 255)),
                    (nose_center,      (0, 165, 255)),
                    (mouth_center,     (255, 255, 0)),
                ]:
                    if c is not None:
                        cv2.circle(frame, c, 3, col, -1)

            if scanning and face_oval_poly:
                face_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(face_mask, [np.array(face_oval_poly, np.int32)], 255)

                def erase_circle(center, radius):
                    if center is None:
                        return
                    cx_i, cy_i = center
                    cv2.circle(face_mask, (cx_i, cy_i), radius, 0, -1)

                erase_circle(left_eye_center,  EXCLUDE_EYE_RADIUS)
                erase_circle(right_eye_center, EXCLUDE_EYE_RADIUS)
                erase_circle(nose_center,      EXCLUDE_NOSE_RADIUS)
                erase_circle(mouth_center,     EXCLUDE_MOUTH_RADIUS)

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
                        cv2.circle(frame, (int(u), int(v)), 1, (0, 255, 0), -1)

                last_scan_points = points_this_frame
                last_scan_frame = frame.copy()

            if not scanning and safe_center is not None and face_center is not None:
                dx = face_center[0] - safe_center[0]
                dy = face_center[1] - safe_center[1]
                dist = math.hypot(dx, dy)
                cv2.circle(frame, safe_center, int(SAFE_RADIUS), (0, 255, 0), 1)
                now_inside_safe = dist <= SAFE_RADIUS
                if was_inside_safe and not now_inside_safe:
                    print("얼굴이 안전 영역을 벗어났습니다. 정지합니다.")
                was_inside_safe = now_inside_safe

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
        try:
            if last_scan_points:
                print(f"[INFO] 스캔 포인트 개수: {len(last_scan_points)}")
                normals = estimate_normals(last_scan_points, neighbor_radius=NEIGHBOR_R_M)
                print(f"[INFO] 법선 추정 완료. (points={len(normals)})")

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
                print("[INFO] scan_points_with_normals.json 저장 완료")

                if last_scan_frame is not None:
                    cv2.imwrite("last_scan_frame.png", last_scan_frame)
                    print("[INFO] last_scan_frame.png 저장 완료")
            else:
                print("[WARN] last_scan_points가 비어 있습니다.")
        except Exception as e:
            print(f"[WARN] 스캔 결과 저장 중 오류: {e}")

        face_mesh.close()
        pipeline.stop()
        cv2.destroyAllWindows()


# =======================
# 5. 스캔 결과를 로봇 좌표로 일괄 변환
# =======================

def convert_scan_points_to_robot(R, t):
    if not os.path.exists("scan_points_with_normals.json"):
        print("[ERROR] scan_points_with_normals.json 이 없습니다. 먼저 스캔을 실행하세요.")
        return

    with open("scan_points_with_normals.json", "r", encoding="utf-8") as f:
        points = json.load(f)
    if len(points) == 0:
        print("[ERROR] 포인트가 없습니다.")
        return

    robot_points = []
    for p in points:
        Xc = p["X_m"]
        Yc = p["Y_m"]
        Zc = p["Z_m"]
        nx_c = p["nx"]
        ny_c = p["ny"]
        nz_c = p["nz"]

        Xr_m, Yr_m, Zr_m = transform_cam_to_robot((Xc, Yc, Zc), R, t)
        Xr_mm = float(round(Xr_m * 1000.0, 1))
        Yr_mm = float(round(Yr_m * 1000.0, 1))
        Zr_mm = float(round(Zr_m * 1000.0, 1))

        nx_r, ny_r, nz_r = transform_normal_cam_to_robot((nx_c, ny_c, nz_c), R)
        rx_rad, ry_rad, rz_rad = normal_to_rpy(nx_r, ny_r, nz_r)

        robot_points.append({
            "x_mm": Xr_mm,
            "y_mm": Yr_mm,
            "z_mm": Zr_mm,
            "nx_r": nx_r,
            "ny_r": ny_r,
            "nz_r": nz_r,
            "rx_rad": rx_rad,
            "ry_rad": ry_rad,
            "rz_rad": rz_rad,
            "pose": [Xr_mm, Yr_mm, Zr_mm, rx_rad, ry_rad, rz_rad],
        })

    with open("scan_points_robot_frame.json", "w", encoding="utf-8") as f:
        json.dump(robot_points, f, ensure_ascii=False, indent=2)
    print("[INFO] scan_points_robot_frame.json 저장 완료 (전체 포인트 로봇 좌표계)")


# =======================
# 메인: 1~5 단계 한 번에 수행
# =======================

if __name__ == "__main__":
    # 0. 로봇 + Realsense 초기화
    robot = RobotInterface()
    pipeline, align, intr = init_realsense()

    # 1~3. Rigid Transform Calibration
    R, t = calibrate_rigid_transform(pipeline, align, intr, robot, num_points=6)

    # 사용 끝난 calibration용 pipeline 정리
    pipeline.stop()
    cv2.destroyAllWindows()

    # 4. 5초 얼굴 스캔 (좌표/법선 카메라 기준)
    scan_face_and_save()

    # 5. 스캔된 모든 포인트를 로봇 좌표계로 변환
    convert_scan_points_to_robot(R, t)

    print("\n[ALL DONE] 캘리브레이션 + 얼굴 스캔 + 로봇 좌표 변환까지 완료.")
