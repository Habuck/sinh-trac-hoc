/*
 * Raspberry Pi 3 bare-metal: nhấp nháy LED ở GPIO24 (chân vật lý 18).
 * GPIO24 thuộc GPFSEL2 (offset 0x08), 3 bit bắt đầu tại bit 12.
 * GPSET0 offset 0x1C (28) bật, GPCLR0 offset 0x28 (40) tắt.
 */

#define GPIO_BASE 0x3F200000
#define GPFSEL2 ((volatile unsigned int *)(GPIO_BASE + 0x08))
#define GPSET0  ((volatile unsigned int *)(GPIO_BASE + 0x1C))
#define GPCLR0  ((volatile unsigned int *)(GPIO_BASE + 0x28))

static void delay(void)
{
    for (volatile unsigned int i = 0x3F0000; i > 0; i--)
        ;
}

void main(void)
{
    *GPFSEL2 = 1u << 12;
    while (1) {
        *GPSET0 = 1u << 24;
        delay();
        *GPCLR0 = 1u << 24;
        delay();
    }
}
