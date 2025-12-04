import cv2
import mediapipe as mp
import numpy as np
import json
from scipy.interpolate import griddata
import math

class FaceInjectionPlanner:
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 제외할 부위의 랜드마크 인덱스 정의
        # 눈, 코, 입 주위를 넉넉하게 잡기 위한 외곽선 인덱스들
        self.EXCLUDE_INDICES = {
            # 왼쪽 눈 윤곽
            'left_eye': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            # 오른쪽 눈 윤곽
            'right_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            # 코 (콧대 및 콧볼 포함 전체 영역)
            'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327], 
            # 입술 (바깥쪽 윤곽)
            'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        }

    def get_landmarks(self, image):
        """이미지에서 3D 랜드마크 추출 (x, y: 픽셀, z: 상대깊이)"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        h, w, c = image.shape
        landmarks = []
        # 첫 번째 얼굴만 사용
        for lm in results.multi_face_landmarks[0].landmark:
            # x, y는 픽셀 좌표로 변환, z는 이미지 너비에 비례하도록 스케일링
            # (MediaPipe의 z는 x와 비슷한 스케일의 상대값임)
            landmarks.append([lm.x * w, lm.y * h, lm.z * w]) 
        
        return np.array(landmarks)

    def create_safety_mask(self, image_shape, landmarks, margin_px=15):
        """
        제외 영역 마스크 생성 (여유 공간 확보)
        margin_px: 제외 영역을 얼마나 더 넓힐지 (픽셀 단위)
        """
        h, w = image_shape[:2]
        # 0: 주사 가능, 255: 제외 영역
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for region_name, indices in self.EXCLUDE_INDICES.items():
            points = landmarks[indices, :2].astype(np.int32)
            # 해당 부위의 볼록 껍질(Convex Hull) 구하기
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)
            
        # Dilate 연산을 통해 제외 영역 확장 (Safety Buffer)
        # "눈, 코, 입 제외한 부분"을 빡빡하게 잡지 않기 위함
        kernel_size = margin_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
            
        return expanded_mask

    def optimize_path_nearest_neighbor(self, points):
        """
        경로 최적화: TSP(외판원 순회) 근사 알고리즘
        현재 위치에서 가장 가까운 점을 찾아 이동하는 방식
        """
        if not points:
            return []
            
        # 1. 시작점 선정 (가장 위쪽, 왼쪽 포인트)
        # y값이 작고 x값이 작은 순서로 정렬하여 첫 번째 점 선택
        start_idx = np.argmin([p['pos'][1] + p['pos'][0]*0.1 for p in points])
        
        optimized_sequence = []
        remaining_indices = list(range(len(points)))
        
        current_idx = remaining_indices.pop(start_idx) # 인덱스 추출 및 리스트에서 제거
        optimized_sequence.append(points[current_idx])
        
        while remaining_indices:
            curr_pos = points[current_idx]['pos']
            
            # 남은 점들 중 가장 가까운 점 찾기 (유클리드 거리)
            # 벡터화 연산을 위해 남은 점들의 좌표 배열 생성
            remaining_points = [points[i]['pos'] for i in remaining_indices]
            remaining_points_np = np.array(remaining_points)
            
            dists = np.linalg.norm(remaining_points_np - curr_pos, axis=1)
            nearest_local_idx = np.argmin(dists)
            
            # 다음 점으로 업데이트
            current_idx = remaining_indices.pop(nearest_local_idx)
            optimized_sequence.append(points[current_idx])
            
        return optimized_sequence

    def process_face_grid(self, image_path, pixel_per_cm=40, output_jsonl='injection_plan.jsonl'):
        """
        전체 프로세스 실행 함수
        pixel_per_cm: 1cm에 해당하는 픽셀 수 (환경에 맞춰 설정 필수)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: 이미지를 읽을 수 없습니다 ({image_path})")
            return

        h, w = image.shape[:2]
        
        # 1. 랜드마크 추출
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            print("Error: 얼굴을 감지하지 못했습니다.")
            return

        # 2. 안전 마스크 생성 (눈코입 + 여유공간)
        # 0.5cm 정도의 여유를 둠 (pixel_per_cm / 2)
        safety_margin = int(pixel_per_cm * 0.5) 
        exclusion_mask = self.create_safety_mask(image.shape, landmarks, margin_px=safety_margin)

        # 3. Grid 생성 및 Z값 보간 (Surface Reconstruction)
        x_min, x_max = int(min(landmarks[:,0])), int(max(landmarks[:,0]))
        y_min, y_max = int(min(landmarks[:,1])), int(max(landmarks[:,1]))
        
        # 1cm 간격 설정
        step = int(pixel_per_cm)
        
        # 격자 좌표 생성
        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max, step),
            np.arange(y_min, y_max, step)
        )
        
        # scipy griddata를 사용하여 Z값 보간 (부드러운 곡면 생성)
        # method='cubic'이 가장 부드럽지만 계산이 느릴 수 있음. 'linear'도 무방.
        points_2d = landmarks[:, :2]
        values_z = landmarks[:, 2]
        
        grid_z = griddata(points_2d, values_z, (grid_x, grid_y), method='linear')

        # 4. 법선 벡터(Normal Vector) 계산 및 유효 포인트 필터링
        # Z맵의 기울기(Gradient) 계산 -> 법선 벡터 유도
        # dy, dx 순서 주의 (numpy gradient는 axis 순서대로 리턴)
        dz_dy, dz_dx = np.gradient(grid_z)
        
        valid_points = []
        face_hull = cv2.convexHull(landmarks[:, :2].astype(np.int32))

        rows, cols = grid_z.shape
        for r in range(rows):
            for c in range(cols):
                px = int(grid_x[r,c])
                py = int(grid_y[r,c])
                pz = grid_z[r,c]
                
                # (1) 보간값이 NaN이 아닌지 확인
                if np.isnan(pz): continue
                
                # (2) 얼굴 외곽선 내부인지 확인 (Face Contour)
                if cv2.pointPolygonTest(face_hull, (px, py), False) < 0:
                    continue
                
                # (3) 제외 영역(눈코입+마진)이 아닌지 확인
                if exclusion_mask[py, px] > 0:
                    continue
                
                # 법선 벡터 계산 로직
                # 기울기: dx, dy
                # 표면의 접평면 벡터: (1, 0, dx), (0, 1, dy)
                # 법선 벡터 = 외적 = (-dx, -dy, 1)
                dx = dz_dx[r, c]
                dy = dz_dy[r, c]
                
                if np.isnan(dx) or np.isnan(dy): continue
                
                # 벡터 정규화 (Normalize)
                normal = np.array([-dx, -dy, 1.0]) # Z방향이 1 (카메라 쪽)
                norm_len = np.linalg.norm(normal)
                if norm_len == 0: continue
                
                normal_unit = normal / norm_len
                
                valid_points.append({
                    'pos': np.array([px, py, pz]),
                    'normal': normal_unit,
                    'grid_idx': (r, c)
                })

        print(f"-> 추출된 주사 포인트 개수: {len(valid_points)}개")

        # 5. 경로 최적화 (Sequence Optimization)
        sorted_points = self.optimize_path_nearest_neighbor(valid_points)

        # 6. 결과 저장 및 시각화
        vis_img = image.copy()
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for idx, pt in enumerate(sorted_points):
                pos = pt['pos']
                normal = pt['normal']
                
                # JSONL 쓰기
                # 주의: 여기서 pos는 픽셀단위(x,y)와 상대깊이(z)입니다.
                # 로봇에 보낼때는 실제 스케일(cm/mm)로 변환이 필요할 수 있습니다.
                record = {
                    "sequence": idx,
                    "target_point": {
                        "x": round(float(pos[0]), 2),
                        "y": round(float(pos[1]), 2),
                        "z": round(float(pos[2]), 2)
                    },
                    "normal_vector": {
                        "nx": round(float(normal[0]), 4),
                        "ny": round(float(normal[1]), 4),
                        "nz": round(float(normal[2]), 4)
                    }
                }
                f.write(json.dumps(record) + "\n")
                
                # 시각화: 점 찍기 (순서대로 색상 변경: 초록 -> 파랑)
                color_ratio = idx / len(sorted_points)
                color = (int(255 * color_ratio), 255 - int(255 * color_ratio), 0) # BGR
                
                cv_pt = (int(pos[0]), int(pos[1]))
                cv2.circle(vis_img, cv_pt, 3, color, -1)
                
                # 시각화: 법선 벡터 (빨간 선)
                # 법선 방향으로 선을 그어 수직인지 확인
                normal_end = (
                    int(pos[0] + normal[0] * 15),
                    int(pos[1] + normal[1] * 15)
                )
                cv2.line(vis_img, cv_pt, normal_end, (0, 0, 255), 1)
                
                # 시작점과 끝점 표시
                if idx == 0:
                    cv2.putText(vis_img, "START", cv_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                elif idx == len(sorted_points) - 1:
                    cv2.putText(vis_img, "END", cv_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # 결과 이미지 저장 (확인용)
        cv2.imwrite("result_visualization.jpg", vis_img)
        cv2.imwrite("mask_visualization.jpg", exclusion_mask)
        
        print(f"-> 완료! 경로가 '{output_jsonl}'에 저장되었습니다.")
        print("-> 시각화 이미지 'result_visualization.jpg'를 확인하여 법선 벡터 방향을 검증하세요.")

# --- 실행부 ---
if __name__ == "__main__":
    # 사용 설정
    # pixel_per_cm 값을 실제 카메라 환경에 맞춰 조정해야 정확한 1cm 간격이 나옵니다.
    # 예: 얼굴 가로 폭이 실제 15cm이고, 이미지상 600픽셀이라면 -> 40 px/cm
    input_image = "face_image.jpg" # 여기에 테스트할 이미지 경로 입력
    
    # 이미지 파일이 없으면 에러가 나므로 try-except 처리 혹은 파일 존재 확인 필요
    import os
    if os.path.exists(input_image):
        planner = FaceInjectionPlanner()
        planner.process_face_grid(input_image, pixel_per_cm=40)
    else:
        print(f"파일을 찾을 수 없습니다: {input_image}")
        print("테스트할 얼굴 이미지 경로를 코드 하단의 input_image 변수에 넣어주세요.")
