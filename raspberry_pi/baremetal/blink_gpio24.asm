; Raspberry Pi 3 bare-metal: nhấp nháy LED ở GPIO24 (chân vật lý 18)
; Cùng cách với slide GPIO18, chỉ đổi thanh ghi chọn chức năng và số bit.
;
; GPIO18: GPFSEL1 (offset 4), bit (18-10)*3 = 24, set/clear bit 18
; GPIO24: GPFSEL2 (offset 8), bit (24-20)*3 = 12, set/clear bit 24
;
; GPSET0 = offset 28 (bật), GPCLR0 = offset 40 (tắt)
; Biên dịch: fasmarm blink_gpio24.asm kernel7.img
; Copy kernel7.img + bootcode.bin + start.elf vào thẻ microSD (FAT32)

format binary as 'img'
org $8000

macro delay {
  local .wait
  mov r2,#0x3F0000
.wait:
  sub r2,#1
  cmp r2,#0
  bne .wait
}

BASE = $3F000000
GPIO_OFFSET = $200000
  mov r0,BASE
  orr r0,GPIO_OFFSET

  mov r1,#1
  lsl r1,#12
  str r1,[r0,#8]      ; finished select GPIO24

loop$:
  mov r1,#1
  lsl r1,#24
  str r1,[r0,#28]     ; 28=LED ON
  delay

  mov r1,#1
  lsl r1,#24
  str r1,[r0,#40]     ; 40=LED OFF
  delay

b loop$
