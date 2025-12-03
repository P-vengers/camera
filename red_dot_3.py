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
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 (환경에 따라 조절 필요)
    lower_red1 = np.array([0,   150, 80]) # V값(밝기) 최소값을 조금 낮춤 (어두운 곳 대비)
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 150, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 잡음 제거
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue # 너무 작은 점 무시
        
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        red_points.append((cx, cy))

    return red_points, mask

# --------------------------
# 2. 메인 루프
# --------------------------
def main():
    log_filename = "calibration_points.jsonl"
    if os.path.exists(log_filename):
        print(f"[주의] 기존 로그 파일이 존재합니다: {log_filename}")
        # os.remove(log_filename) # 필요시 주석 해제하여 초기화

    # Realsense 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ★ 중요: Depth를 Color에 맞춤
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    # ★ 중요: Align을 했으므로 'Color' 스트림의 Intrinsics를 가져와야 함
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print(f"[INFO] Intrinsics Loaded: {intrinsics.width}x{intrinsics.height}, fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print("[사용법]")
    print(" 1. 빨간 마커가 화면에 인식되는지 확인 (초록 원 표시)")
    print(" 2. 화면 중앙에 가장 가까운 마커 하나만 인식되도록 조정")
    print(" 3. 's' 키를 누르면 현재 좌표 저장")
    print(" 4. 'q' 키를 누르면 종료")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            
            # 빨간점 탐지
            red_points, mask = find_red_markers(frame)

            # 화면 중앙 좌표
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            target_point = None
            min_dist = float('inf')

            # 여러 개가 잡히면 화면 중앙에 가장 가까운 놈 하나만 타겟으로 선정
            for (u, v) in red_points:
                dist = (u - center_x)**2 + (v - center_y)**2
                if dist < min_dist:
                    min_dist = dist
                    target_point = (u, v)

            # 타겟이 있으면 정보 계산
            current_data = None
            
            if target_point:
                u, v = target_point
                
                # 깊이 추출
                depth_dist = depth_frame.get_distance(u, v)
                
                if 0.1 < depth_dist < 2.0: # 유효 거리 필터 (10cm ~ 2m)
                    # ★ 중요: SDK 함수 사용하여 3D 좌표 변환 (왜곡 보정 포함)
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_dist)
                    
                    X_m, Y_m, Z_m = point_3d[0], point_3d[1], point_3d[2]

                    current_data = {
                        "u": int(u), "v": int(v),
                        "depth_m": round(depth_dist, 4),
                        "X_m": round(X_m, 4),
                        "Y_m": round(Y_m, 4),
                        "Z_m": round(Z_m, 4)
                    }

                    # 화면 표시
                    cv2.circle(frame, (u, v), 10, (0, 255, 0), 2) # 타겟 원
                    text = f"X:{X_m:.3f} Y:{Y_m:.3f} Z:{Z_m:.3f}"
                    cv2.putText(frame, text, (u - 50, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 중앙 십자선 표시 (조준 도움용)
            cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 0), 1)
            cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 0), 1)

            cv2.imshow("Calibration Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 's' 키를 누르면 저장
            if key == ord('s'):
                if current_data:
                    timestamp = time.time()
                    log_entry = {"timestamp": timestamp, "markers": [current_data]}
                    
                    with open(log_filename, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                    
                    print(f"[저장됨] {current_data}")
                    # 시각적 피드백 (화면 깜빡임 효과)
                    cv2.putText(frame, "SAVED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imshow("Calibration Collector", frame)
                    cv2.waitKey(200) # 0.2초 대기
                else:
                    print("[실패] 마커가 감지되지 않았거나 깊이값이 유효하지 않습니다.")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
