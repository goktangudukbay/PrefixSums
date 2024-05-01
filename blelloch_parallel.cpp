#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <iostream>


#define POWER 22
#define SIZE (int)pow(2, POWER)

static inline int powerof2(int a){
   return (1 << a);
}

int main(int argc, char *argv[])
{
    int* numbers = (int*)malloc(sizeof(int)*SIZE);
    for(int i = 0; i < SIZE; i++)
        numbers[i] = i;

    double start = omp_get_wtime();
    //up
    for(int i = 0; i < POWER; i++){
    #pragma omp parallel for num_threads(32)
        for(int j = 0; j < SIZE; j += powerof2(i+1)){
            numbers[j + powerof2(i+1) - 1] += numbers[j + powerof2(i) - 1];
        }
    }

    //clear
    numbers[SIZE-1] = 0;

    //down
    for(int i = POWER - 1; i >= 0; i--){
    
        #pragma omp parallel for num_threads(32)

        for(int j = 0; j < SIZE; j += powerof2(i+1)){
            int i2 = powerof2(i);
            int i21 = powerof2(i+1);

            int temp = numbers[j + i2 - 1];
            numbers[j + i2 - 1] = numbers[j + i21 - 1];
            numbers[j + i21 - 1] = temp + numbers[j + i21 - 1];
        }
    }

    double end = omp_get_wtime();
    printf("Execution Time: %f microseconds\n", 1000000*(end - start));

    free(numbers);
    return 0;
}