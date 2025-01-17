import cv2
import numpy as np
from ultralytics import YOLO
import time
from gtts import gTTS  # gTTS 라이브러리 추가
from scipy.spatial import distance
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# YOLOv8 모델 로드
model = YOLO('best .pt')  # 전체 객체 탐지 모델
# 웹캠 초기화
cap = cv2.VideoCapture(2)
# ROI 영역 정의
jig_positions = {
    1: ((250, 70), (370, 190)),   # 지그 1
    2: ((380, 70), (505, 190)),   # 지그 2
    3: ((515, 70), (630, 190))    # 지그 3
}
siling_roi = ((510, 300), (630, 410))  # 실링 검출용 ROI
frame_interval = 0.1  # 0.1초 간격으로 추론
last_time = time.time()
last_state = {"siling_detected": False, "distance_state": None}
jig_status = {1: 0, 2: 0, 3: 0}
empty_jig_ids = [1, 2, 3]

# TTS 비동기 함수
async def async_speak(message, key):
    """TTS를 비동기로 처리하여 다른 연산이 중단되지 않도록 함"""
    if last_state.get(key) != message:
        last_state[key] = message
        asyncio.create_task(run_tts(message))

async def run_tts(message):
    """TTS 실행"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, execute_tts, message)
    except Exception as e:
        print(f"음성 출력 오류: {e}")

def execute_tts(message):
    """TTS 처리"""
    tts = gTTS(text=message, lang='ko')
    tts.save("alert.mp3")
    os.system("mpg123 alert.mp3")

# ROI 확인 함수
def check_jig_position(mask_points, roi):
    """ROI 내 객체 위치 확인"""
    (x1, y1), (x2, y2) = roi
    return any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for point in mask_points)

# 비동기 YOLO 추론 함수
async def async_yolo_processing(frame):
    """YOLO 추론 비동기 처리"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        results = await loop.run_in_executor(pool, model.predict, frame, False)
    return results

# 비동기 거리 계산 함수
async def async_calculate_distance(person_masks, robot_arm_masks):
    """비동기로 거리 계산 및 경고 메시지 출력"""
    if person_masks and robot_arm_masks:
        loop = asyncio.get_event_loop()
        min_distance = float('inf')

        def calculate():
            nonlocal min_distance
            for person_mask in person_masks:
                for robot_mask in robot_arm_masks:
                    dist = distance.cdist(person_mask, robot_mask).min()
                    min_distance = min(min_distance, dist)
            return min_distance

        min_distance = await loop.run_in_executor(None, calculate)
        print(f"Closest distance between 'person' and 'robot_arm': {min_distance:.2f} pixels")

        if min_distance <= 5:
            await async_speak("정지하겠습니다.", "distance_state")
        elif min_distance <= 40:
            await async_speak("위험합니다. 뒤로 가세요.", "distance_state")
        else:
            last_state["distance_state"] = None
    else:
        last_state["distance_state"] = None

# 메인 비동기 함수
async def main():
    global last_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - last_time >= frame_interval:
            last_time = time.time()
            results = await async_yolo_processing(frame)
            if results and results[0].masks:
                detected_objects = {"person": [], "robot_arm": [], "siling": [], "holder": [], "ice_cream": []}
                for i, mask in enumerate(results[0].masks.xy):
                    cls = int(results[0].boxes.cls[i])
                    conf = results[0].boxes.conf[i]
                    label = model.names[cls]
                    if conf < 0.4:
                        continue
                    if label in detected_objects:
                        mask_points = np.array(mask, dtype=np.int32)
                        detected_objects[label].append(mask_points)
                        color = {
                            "person": (0, 255, 0), "robot_arm": (255, 0, 0),
                            "siling": (0, 0, 255), "holder": (255, 255, 0),
                            "ice_cream": (255, 0, 255)
                        }.get(label, (255, 255, 255))
                        x_center = int(np.mean(mask_points[:, 0]))
                        y_center = int(np.mean(mask_points[:, 1]))
                        cv2.polylines(frame, [mask_points], isClosed=True, color=color, thickness=2)
                        cv2.putText(frame, label, (x_center, y_center - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                siling_detected = any(
                    check_jig_position(mask_points, siling_roi)
                    for mask_points in detected_objects["siling"]
                )
                if siling_detected:
                    print("실링 검출: 실링을 제거해주세요!")
                    await async_speak("실링을 제거해주세요!", "siling_detected")
                else:
                    last_state["siling_detected"] = False

                for jig_id, roi in jig_positions.items():
                    jig_empty = not any(
                        check_jig_position(mask_points, roi)
                        for label in ['holder', 'ice_cream']
                        for mask_points in detected_objects.get(label, [])
                    )
                    if not jig_empty:
                        jig_status[jig_id] += frame_interval
                    else:
                        jig_status[jig_id] = 0
                    if jig_status[jig_id] >= 3.0:
                        if jig_id in empty_jig_ids:
                            empty_jig_ids.remove(jig_id)
                            print(f"Jig {jig_id} removed from empty list.")
                    else:
                        if jig_id not in empty_jig_ids:
                            empty_jig_ids.append(jig_id)
                print(f"Current empty jig IDs: {empty_jig_ids}")

                person_masks = detected_objects["person"]
                robot_arm_masks = detected_objects["robot_arm"]
                await async_calculate_distance(person_masks, robot_arm_masks)

                for jig_id, ((x1, y1), (x2, y2)) in jig_positions.items():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, f"Jig {jig_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                (x1, y1), (x2, y2) = siling_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Siling ROI", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow('YOLOv8 Segmentation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 비동기 실행
asyncio.run(main())
