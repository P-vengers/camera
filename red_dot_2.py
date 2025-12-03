import cv2
import numpy as np
import pyrealsense2 as rs
import json
import time
import os

# --------------------------
# 1. 빨간 마커 탐지 (HSV 마스크)
# --------------------------
def find_red_markers(frame_bgr):
    """
    BGR 이미지를 입력받아,
    HSV 마스크로 빨간색 영역을 찾아
    각 덩어리(마커)의 중심 좌표 리스트를 반환.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 (필요하면 튜닝)
    lower_red1 = np.array([0,   150, 120])
    upper_red1 = np.array([8,   255, 255])
    lower_red2 = np.array([172, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 잡음 제거용 morphological operation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 컨투어(빨간 영역) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:   # 마커 크기에 따라 20~50 정도로 조절
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        red_points.append((cx, cy))

    return red_points, mask

# --------------------------
# 2. 메인 루프
# --------------------------
def main():
    # JSONL 로그 파일 이름
    log_filename = "red_markers_log.jsonl"
    print(f"[INFO] 로그 파일: {os.path.abspath(log_filename)}")

    # ==============
    # RealSense 초기화
    # ==============
    pipeline = rs.pipeline()
    config = rs.config()

    # 컬러 + 뎁스 스트림 활성화
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # depth를 color에 align
    align_to = rs.stream.color
    align = rs.align(align_to)

    # depth intrinsics 가져오기 (3D 계산용)
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f"[INFO] Depth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    try:
        while True:
            # 1) 프레임 받기
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # BGR
            h, w, _ = frame.shape

            # 2) 빨간점 탐지
            red_points, red_mask = find_red_markers(frame)

            markers_payload = []

            for (u, v) in red_points:
                if not (0 <= u < w and 0 <= v < h):
                    continue

                depth_m = float(depth_frame.get_distance(u, v))
                if depth_m <= 0:
                    continue

                # 카메라 기준 3D (meter)
                Z_m = depth_m
                X_m = (u - cx) / fx * Z_m
                Y_m = (v - cy) / fy * Z_m

                # 시각화: 초록색 동그라미 + depth 표시
                cv2.circle(frame, (u, v), 7, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"{depth_m:.3f}m",
                    (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                markers_payload.append({
                    "u": int(u),
                    "v": int(v),
                    "depth_m": depth_m,
                    "X_m": X_m,
                    "Y_m": Y_m,
                    "Z_m": Z_m,
                })

            # 3) JSONL로 저장 (프레임 단위)
            if markers_payload:
                payload = {
                    "timestamp": time.time(),
                    "markers": markers_payload,
                }
                # 콘솔에도 출력
                print(json.dumps(payload), flush=True)

                # 파일에 한 줄 append (jsonl 형식)
                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")

            # 4) 화면 출력
            cv2.putText(
                frame,
                "Red marker detection. Press 'q' to quit.",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("D435 Red Marker Detection", frame)
            # 디버깅용
            # cv2.imshow("Red Mask", red_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
