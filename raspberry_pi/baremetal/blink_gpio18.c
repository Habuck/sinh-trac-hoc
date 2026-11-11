/*
 * Raspberry Pi 3 bare-metal: nhấp nháy LED ở GPIO18 (Pin 12)
 * Cần thêm startup code (đặt stack, gọi main) để tạo kernel7.img.
 */

#define GPIO_BASE 0x3F200000
#define GPFSEL1 ((volatile unsigned int *)(GPIO_BASE + 0x04))
#define GPSET0  ((volatile unsigned int *)(GPIO_BASE + 0x1C))
#define GPCLR0  ((volatile unsigned int *)(GPIO_BASE + 0x28))

static void delay(void)
{
    for (volatile unsigned int i = 0x3F0000; i > 0; i--)
        ;
}

void main(void)
{
    *GPFSEL1 = 1 << 24;
    while (1) {
        *GPSET0 = 1 << 18;
        delay();
        *GPCLR0 = 1 << 18;
        delay();
    }
}
