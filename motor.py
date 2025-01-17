import RPi.GPIO as GPIO
import time
import curses  # 터미널 기반 키 입력 처리 라이브러리
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 헤드리스 환경에 맞는 백엔드 설정 (그래프 파일 출력)

# GPIO 핀 설정
SERVO_PIN1 = 17  # 모터1 연결 핀
SERVO_PIN2 = 19  # 모터2 연결 핀

# PID 클래스 정의
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def calculate(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

# ServoMotor 클래스 정의
class ServoMotor:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN1, GPIO.OUT)
        GPIO.setup(SERVO_PIN2, GPIO.OUT)

        self.pwm1 = GPIO.PWM(SERVO_PIN1, 50)
        self.pwm2 = GPIO.PWM(SERVO_PIN2, 50)
        self.pwm1.start(0)
        self.pwm2.start(0)

        self.angle1 = 0
        self.angle2 = 104
        self.target_angle1 = 0
        self.target_angle2 = 104
        self.set_angle(self.angle1, self.angle2)

    def set_angle(self, angle1, angle2):
        duty1 = 2.5 + (angle1 / 18)
        duty2 = 2.5 + (angle2 / 18)
        self.pwm1.ChangeDutyCycle(duty1)
        self.pwm2.ChangeDutyCycle(duty2)
        time.sleep(0.05)

        self.angle1 = angle1
        self.angle2 = angle2

    def cleanup(self):
        self.pwm1.stop()
        self.pwm2.stop()
        GPIO.cleanup()

def main(stdscr):
    servo = ServoMotor()
    
    # PID 값 조정
    pid1 = PIDController(0.6, 0.005, 0.3)  # 최적화된 PID 값
    pid2 = PIDController(0.6, 0.005, 0.3)

    curses.cbreak()
    stdscr.nodelay(1)
    stdscr.keypad(1)

    stdscr.addstr("PID 제어: 'i' (위), 'k' (아래), 'j' (왼쪽), 'l' (오른쪽), 'q' (종료)\n")

    time_data = []
    angle1_data = []
    target1_data = []
    start_time = time.time()

    try:
        while True:
            output1 = pid1.calculate(servo.target_angle1, servo.angle1)
            output2 = pid2.calculate(servo.target_angle2, servo.angle2)

            new_angle1 = max(-40, min(80, servo.angle1 + output1))
            new_angle2 = max(90, min(150, servo.angle2 + output2))

            servo.set_angle(new_angle1, new_angle2)

            current_time = time.time() - start_time
            time_data.append(current_time)
            angle1_data.append(servo.angle1)
            target1_data.append(servo.target_angle1)

            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('j'):
                servo.target_angle1 = max(-40, servo.target_angle1 - 2)
            elif key == ord('l'):
                servo.target_angle1 = min(80, servo.target_angle1 + 2)
            elif key == ord('i'):
                servo.target_angle2 = min(150, servo.target_angle2 + 2)
            elif key == ord('k'):
                servo.target_angle2 = max(90, servo.target_angle2 - 2)

            stdscr.addstr(f"\r모터1 목표 각도: {servo.target_angle1}, 모터2 목표 각도: {servo.target_angle2}")
            stdscr.refresh()
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        servo.cleanup()
        curses.endwin()

        # 그래프 저장
        plt.figure()
        plt.plot(time_data, target1_data, label='Target Angle (모터1)')
        plt.plot(time_data, angle1_data, label='Current Angle (모터1)', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('PID Control: Target vs Current Angle')
        plt.legend()
        plt.grid()
        plt.savefig('pid_control_graph.png')  # 그래프 파일로 저장
        print("그래프가 'pid_control_graph.png' 파일로 저장되었습니다.")

if __name__ == "__main__":
    curses.wrapper(main)

from PIL import Image
img = Image.open('pid_control_graph.png')
img.show()
plt.savefig('/home/pid/pid_control_graph.png')

