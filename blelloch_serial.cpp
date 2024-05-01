#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>

#define SIZE 4194304

int main(int argc, char *argv[])
{
    printf("SIZE %d \n", SIZE);
    int* numbers = (int*)malloc(sizeof(int)*SIZE);
    for(int i = 0; i < SIZE; i++)
        numbers[i] = i;

    auto start = std::chrono::system_clock::now();
    for(int i = 1; i < SIZE; i++)
        numbers[i] = numbers[i] + numbers[i-1];

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Total Number of Elements (N): " << SIZE << "\n Time: " << elapsed.count() << std::endl;

    free(numbers);
    return 0;
}