
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASKA_W  3
#define RADIUS MASKA_W/2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + MASKA_W - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))



#define MAX_IMAGE_WIDTH  3840
#define MAX_IMAGEHEIGHT  2160
#define MAX_GRAY_LEVEL  255      
#define GRAYLEVEL       256      
#define MAX_FILENAME    256      
#define MAX_BUFFERSIZE  256


float image1[MAX_IMAGE_WIDTH][MAX_IMAGEHEIGHT],
image2[MAX_IMAGE_WIDTH][MAX_IMAGEHEIGHT];
int x_size1, y_size1, x_size2, y_size2;




void load_image_data()
{
    char file_name[MAX_FILENAME];
    char buffer[MAX_BUFFERSIZE];
    FILE* fp;
    int max_gray;
    int x, y;


    //printf("<Файл должен быть в формате .pgm>\n\n");
    printf("Name? (*.pgm) : ");
    scanf("%s", file_name);
    fp = fopen(file_name, "rb");
    if (NULL == fp) {
        printf(" not such file in directory!\n\n");
        exit(1);
    }
    /* Check of file-type ---P5 */
    fgets(buffer, MAX_BUFFERSIZE, fp);
    if (buffer[0] != 'P' || buffer[1] != '5') {
        printf(" Wrong format (нужен .pgm)!\n\n");
        exit(1);
    }
    /* input of x_size1, y_size1 */
    x_size1 = 0;
    y_size1 = 0;
    while (x_size1 == 0 || y_size1 == 0) {
        fgets(buffer, MAX_BUFFERSIZE, fp);
        if (buffer[0] != '#') {
            sscanf(buffer, "%d %d", &x_size1, &y_size1);
        }
    }
    /* input of max_gray */
    max_gray = 0;
    while (max_gray == 0) {
        fgets(buffer, MAX_BUFFERSIZE, fp);
        if (buffer[0] != '#') {
            sscanf(buffer, "%d", &max_gray);
        }
    }
    /* Display of parameters */

    if (x_size1 > MAX_IMAGE_WIDTH || y_size1 > MAX_IMAGEHEIGHT) {
        printf("     wrong size, need - %d x %d\n\n", MAX_IMAGE_WIDTH, MAX_IMAGEHEIGHT);
        exit(1);
    }
    if (max_gray != MAX_GRAY_LEVEL) {
        printf("    yarko!\n\n");
        exit(1);
    }
    /* Input of image data*/
    for (y = 0; y < y_size1; y++) {
        for (x = 0; x < x_size1; x++) {
            image1[y][x] = (unsigned char)fgetc(fp);
        }
    }
    printf("----Succes-----\n\n");
    fclose(fp);
}




void save_image_data()
{
    char file_name[MAX_FILENAME];
    FILE* fp; /* File pointer */
    int x, y; /* Loop variable */

    /* Output file open */


    printf("<format file .pgm>\n\n");
    printf("Name output file? (*.pgm) : ");
    scanf("%s", file_name);
    fp = fopen(file_name, "wb");
    /* output of pgm file header information */
    fputs("P5\n", fp);
    fputs("# Created by Image Processing\n", fp);
    fprintf(fp, "%d %d\n", x_size2, y_size2);
    fprintf(fp, "%d\n", MAX_GRAY_LEVEL);
    /* Output of image data */
    for (y = 0; y < y_size2; y++) {
        for (x = 0; x < x_size2; x++) {
            fputc(image2[y][x], fp);
        }
    }
    printf("\n-----Image data output OK-----\n\n");

    fclose(fp);
}



void load_image_file(char* filename)
{
    char buffer[MAX_BUFFERSIZE];
    FILE* fp;
    int max_gray;
    int x, y;

    /* Input file open */
    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("     The file doesn't exist!\n\n");
        exit(1);
    }
    /* Check of file-type ---P5 */
    fgets(buffer, MAX_BUFFERSIZE, fp);
    if (buffer[0] != 'P' || buffer[1] != '5') {
        printf("     Mistaken file format, not P5!\n\n");
        exit(1);
    }
    /* input of x_size1, y_size1 */
    x_size1 = 0;
    y_size1 = 0;
    while (x_size1 == 0 || y_size1 == 0) {
        fgets(buffer, MAX_BUFFERSIZE, fp);
        if (buffer[0] != '#') {
            sscanf(buffer, "%d %d", &x_size1, &y_size1);
        }
    }
    /* input of max_gray */
    max_gray = 0;
    while (max_gray == 0) {
        fgets(buffer, MAX_BUFFERSIZE, fp);
        if (buffer[0] != '#') {
            sscanf(buffer, "%d", &max_gray);
        }
    }
    if (x_size1 > MAX_IMAGE_WIDTH || y_size1 > MAX_IMAGEHEIGHT) {
        printf("     Image size exceeds %d x %d\n\n",
            MAX_IMAGE_WIDTH, MAX_IMAGEHEIGHT);
        printf("     Please use smaller images!\n\n");
        exit(1);
    }
    if (max_gray != MAX_GRAY_LEVEL) {
        printf("     Invalid value of maximum gray level!\n\n");
        exit(1);
    }
    /* Input of image data*/
    for (y = 0; y < y_size1; y++) {
        for (x = 0; x < x_size1; x++) {
            image1[y][x] = (float)fgetc(fp);
        }
    }
    fclose(fp);
}




void save_image_file(char* filename)
{
    FILE* point_file;
    int x, y;

    point_file = fopen(filename, "wb");

    fputs("P\n", point_file);
    fputs("Processing\n", point_file);
    fprintf(point_file, "%d %d\n", x_size2, y_size2);
    fprintf(point_file, "%d\n", MAX_GRAY_LEVEL);
    /* Output of image data */
    for (y = 0; y < y_size2; y++) {
        for (x = 0; x < x_size2; x++) {
            fputc(image2[y][x], point_file);
        }
    }
    fclose(point_file);
}




__global__ void svertka(float* I, const float* __restrict__ M, float* P, int width, int height) {
    __shared__ float N_ds[w][w];


    // Загрузка первой партии
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x, destY = dest / w, destX = dest % w,
        srcY = blockIdx.y * TILE_WIDTH + destY - RADIUS, srcX = blockIdx.x * TILE_WIDTH + destX - RADIUS,
        src = srcY * width + srcX;

    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
    {
        N_ds[destY][destX] = I[src];
    }
    else
    {
        N_ds[destY][destX] = 0;
    }

    for (int i = 1; i <= (w * w) / (TILE_WIDTH * TILE_WIDTH); i++)
    {
        // Загрузка вторйо партии
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w, destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - RADIUS;
        srcX = blockIdx.x * TILE_WIDTH + destX - RADIUS;
        src = srcY * width + srcX;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float iac_cum = 0;
    int y, x;
    for (y = 0; y < MASKA_W; y++)
    {
        for (x = 0; x < MASKA_W; x++)
        {
            iac_cum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASKA_W + x];
        }

    }


    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width)
        P[y * width + x] = iac_cum;

    __syncthreads();

}

void sobel_filtering()
{

    float weight[3][3] = { { -1,  0,  1 },
                           { -2,  0,  2 },
                           { -1,  0,  1 } };
    float pixel_value;

    int x, y, i, j;
    float* Image_input_from_device_point;
    float* Image_output_from_device_poin;
    float* deviceMask;

    cudaMalloc((void**)&Image_input_from_device_point, x_size1 * y_size1 * sizeof(float));
    cudaMalloc((void**)&Image_output_from_device_poin, x_size1 * y_size1 * sizeof(float));
    cudaMalloc((void**)&deviceMask, 3 * 3 * sizeof(float));

    cudaMemcpy(Image_input_from_device_point, image1, x_size1 * y_size1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, weight, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);



    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 0; y < y_size2; y++)
    {
        for (x = 0; x < x_size2; x++)
        {
            image2[y][x] = 0;
        }
    }

    dim3 dimGrid(ceil((float)x_size1 / TILE_WIDTH), ceil((float)y_size1 / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    svertka << <dimGrid, dimBlock >> > (Image_input_from_device_point, deviceMask, Image_output_from_device_poin, x_size1, y_size1);


    cudaMemcpy(image2, Image_output_from_device_poin, x_size2 * y_size2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Image_input_from_device_point);
    cudaFree(Image_output_from_device_poin);
    cudaFree(deviceMask);

}


int main()
{

    
    load_image_data();

    clock_t begin = clock();
    sobel_filtering();   //Приминение фильтра собеля
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n\nTime: %f\n", time_spent);
    save_image_data();
    return 0;
}