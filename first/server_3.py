import socket
import numpy as np
import json

# =========================================================
# [STEP 1] 여기에 캘리브레이션으로 구한 3x4 행렬을 넣으세요!
# =========================================================
TRANSFORMATION_MATRIX = np.array([
    [-0.99781501, -0.00102843, 0.00152802, 400.06836136],
    [0.00446322, 0.99034210, -0.02868832, 40.26705288],
    [0.01027314, -0.02655038, -0.98431808, 409.87350826],
])
# =========================================================

DEFAULT_ORI = [148.29, -179.06, -61.2]

HOST = "0.0.0.0"
PORT = 200

# ---------------------------------------------------------
# JSONL 파일에서 1줄씩 읽기 위한 제너레이터
# ---------------------------------------------------------
def jsonl_reader(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# 파일 제너레이터 준비
data_iter = jsonl_reader("calibration_data_final.jsonl")

# ---------------------------------------------------------
# 카메라(m) → 로봇(mm) 변환
# ---------------------------------------------------------
def calculate_robot_pos(cam_x, cam_y, cam_z):
    cam_vec = np.array([cam_x * 1000, cam_y * 1000, cam_z * 1000, 1.0])
    robot_pos = np.dot(TRANSFORMATION_MATRIX, cam_vec)
    return robot_pos

# ---------------------------------------------------------
# 서버
# ---------------------------------------------------------
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("\n=========================================")
    print(f"[SERVER] 수동 검증 서버 시작 (PORT: {PORT})")
    print("[INFO] 로봇이 'shot'을 보내면 JSONL 파일에서 좌표를 자동으로 읽어 전송합니다.")
    print("=========================================\n")

    conn, addr = server.accept()
    print(f"[SERVER] 로봇 접속됨 → {addr}")

    global data_iter

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break

            msg = data.decode().strip()
            print(f"\n[FROM ROBOT] 수신된 메시지: {msg}")

            # 로봇이 좌표 요청
            if msg == "shot":
                try:
                    # JSONL 한 줄 읽기
                    entry = next(data_iter)

                    cx = float(entry["X_m"])
                    cy = float(entry["Y_m"])
                    cz = float(entry["Z_m"])

                    # 변환
                    rx, ry, rz = calculate_robot_pos(cx, cy, cz)

                    # 소수점 6자리 제한
                    cx_fmt = f"{cx:.6f}"
                    cy_fmt = f"{cy:.6f}"
                    cz_fmt = f"{cz:.6f}"

                    # 출력
                    print("---------------------------------")
                    print(f"📂 JSONL Camera (m): {cx_fmt}, {cy_fmt}, {cz_fmt}")
                    print(f"🤖 변환 (Robot mm): {rx:.2f}, {ry:.2f}, {rz:.2f}")
                    print("---------------------------------")

                    # 로봇 전송 패킷
                    send_str = f"{rx:.2f},{ry:.2f},{rz:.2f},{DEFAULT_ORI[0]},{DEFAULT_ORI[1]},{DEFAULT_ORI[2]}"

                    conn.sendall((send_str + "\r\n").encode())
                    print(f"[TO ROBOT] 전송 완료 → {send_str}")

                except StopIteration:
                    print("⚠️ JSONL 파일에 더 이상 데이터가 없습니다!")
                    conn.sendall(("EOF\r\n").encode())

        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
            break

    conn.close()
    server.close()


if __name__ == "__main__":
    start_server()
