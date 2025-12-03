import cv2
import numpy as np
import pyrealsense2 as rs
import json
import time
import os

# 전역 변수: 마우스 클릭 좌표
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"[클릭] 좌표: ({x}, {y})")

def main():
    log_filename = "calibration_data_final.jsonl"
    
    # 1. RealSense 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Align (Depth -> Color)
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    # Intrinsics 가져오기 (Color 기준)
    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    
    print("=== 캘리브레이션 데이터 수집기 (고정 카메라용) ===")
    print("1. 카메라를 고정하세요.")
    print("2. 화면에 보이는 빨간 점(마커)을 마우스로 클릭하세요.")
    print("3. 좌표가 출력되면, 로봇을 해당 위치로 이동시키고 로봇 좌표를 따로 기록하세요.")
    print("4. 'q'를 눌러 종료합니다.")

    cv2.namedWindow("Calibration View")
    cv2.setMouseCallback("Calibration View", mouse_callback)

    global clicked_point

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue
            
            frame = np.asanyarray(color_frame.get_data())
            
            # 안내 문구
            cv2.putText(frame, "Click Red Marker on Screen", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 마우스 클릭 처리
            if clicked_point is not None:
                u, v = clicked_point
                
                # 클릭한 곳의 깊이 확인
                depth_dist = depth_frame.get_distance(u, v)
                
                if depth_dist > 0:
                    # 3D 변환 (Deproject)
                    point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth_dist)
                    X_m, Y_m, Z_m = point_3d
                    
                    # 화면에 표시
                    cv2.circle(frame, (u, v), 5, (0, 0, 255), -1) # 클릭 지점 빨간점
                    cv2.circle(frame, (u, v), 10, (0, 255, 0), 2) # 초록 테두리
                    
                    info_text = f"X:{X_m:.3f} Y:{Y_m:.3f} Z:{Z_m:.3f}"
                    cv2.putText(frame, info_text, (u + 15, v), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    print(f"\n✅ [저장됨] Pixel:({u},{v}) -> Camera 3D(m): {X_m:.4f}, {Y_m:.4f}, {Z_m:.4f}")
                    print("👉 이 점에 로봇을 갖다 대고 로봇 좌표를 기록하세요!")

                    # 파일 저장
                    data = {
                        "timestamp": time.time(),
                        "u": u, "v": v,
                        "X_m": X_m, "Y_m": Y_m, "Z_m": Z_m
                    }
                    with open(log_filename, "a") as f:
                        f.write(json.dumps(data) + "\n")
                    
                    # 클릭 상태 초기화 (중복 저장 방지)
                    clicked_point = None
                    
                else:
                    print("⚠️ 깊이 값 측정 불가 (거리가 너무 가깝거나 멉니다). 다시 클릭하세요.")
                    clicked_point = None

            cv2.imshow("Calibration View", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
