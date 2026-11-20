"""LED GPIO24 (chân vật lý 18): sáng rồi tắt. Chấm kết quả, không cần assembly."""
from time import sleep
from gpiozero import LED

led = LED(24)

while True:
    led.on()    # LED sáng
    sleep(0.5)
    led.off()   # LED tắt
    sleep(0.5)
