/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"

void main(void){

    /* stop watchdog timer */
     WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

    uint16_t i, j, m, lr = 6;

    for (i = 0; i < 16; i ++){
            memset(KERNEL_GRADIENT, 0, sizeof(dtype) * 320);
            memset(BIAS_GRADIENT, 0, sizeof(dtype) * 10);
            LOSS = 0;

            for (j = 0; j < 128; j ++){

                memcpy(&TARGET, &Y[j], 1);

                for (m = 0; m < 32; m ++){
                    INPUT[m << 1] = X[(j<<5) + m];
                    INPUT_T[m] = X[(j<<5) + m];
                }

                dense(&OUTPUT_MAT, &INPUT_MAT, &KERNEL_MAT, &BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);

                softmax(&ACTIVATION_MAT, &OUTPUT_MAT, FIXED_POINT_PRECISION, 7);

                LOSS += (cce_loss(&ACTIVATION_MAT, TARGET, FIXED_POINT_PRECISION) >> 4);

                cce_kernel_gradient(&KERNEL_GRADIENT_TEMP_MAT, &ACTIVATION_MAT, &INPUT_T_MAT, TARGET, 4, FIXED_POINT_PRECISION);
                cce_bias_gradient(&BIAS_GRADIENT_TEMP_MAT, &ACTIVATION_MAT, TARGET, 4, FIXED_POINT_PRECISION);

                matrix_add(&KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_TEMP_MAT);
                matrix_add(&BIAS_GRADIENT_MAT, &BIAS_GRADIENT_MAT, &BIAS_GRADIENT_TEMP_MAT);

            }

            matrix_neg(&KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_MAT, FIXED_POINT_PRECISION);
            matrix_neg(&BIAS_GRADIENT_MAT, &BIAS_GRADIENT_MAT, FIXED_POINT_PRECISION);

            gradient_descent(&KERNEL_MAT, &KERNEL_GRADIENT_MAT, &BIAS_MAT, &BIAS_GRADIENT_MAT, lr);
            if (i == 2 || i == 8 || i == 12 || i == 16) lr ++;
            LOSS = LOSS >> 3;
            __no_operation();

        }

        for (j = 128; j < 160; j ++){

            memcpy(&TARGET, &Y[j], 1);

            for (m = 0; m < 32; m ++){
                INPUT[m << 1] = X[(j<<5) + m];
                INPUT_T[m] = X[(j<<5) + m];
            }

            dense(&OUTPUT_MAT, &INPUT_MAT, &KERNEL_MAT, &BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);

            LABEL = argmax(&OUTPUT_MAT);
            __no_operation();
        }

    __no_operation();
}
