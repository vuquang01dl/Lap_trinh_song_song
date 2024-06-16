#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

using namespace cv;

void convert_to_grayscale(Mat& src, Mat& dst, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            Vec3b color = src.at<Vec3b>(i, j);
            uchar gray = (uchar)(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]);
            dst.at<uchar>(i, j) = gray;
        }
    }
}

void adjust_brightness(Mat& img, int brightness) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                img.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(img.at<Vec3b>(i, j)[c] + brightness);
            }
        }
    }
}

void adjust_contrast(Mat& img, double alpha) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                img.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(img.at<Vec3b>(i, j)[c] * alpha);
            }
        }
    }
}

void calculate_histogram(Mat& img, int* histogram) {
    for (int i = 0; i < 256; ++i)
        histogram[i] = 0;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            uchar pixel = img.at<uchar>(i, j);
            histogram[pixel]++;
        }
    }
}

void print_histogram(int* histogram) {
    for (int i = 0; i < 256; ++i) {
        printf("%d: %d\n", i, histogram[i]);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    char* input_image = argv[1];
    char* output_image = argv[2];
    int brightness = 50;  // Brightness adjustment value
    double contrast = 1.2; // Contrast adjustment value

    Mat img, gray_img;
    int rows_per_process, remainder_rows;
    int img_rows, img_cols;

    double start_time, end_time;

    if (rank == 0) {
        img = imread(input_image);
        if (img.empty()) {
            printf("Could not open or find the image: %s\n", input_image);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        img_rows = img.rows;
        img_cols = img.cols;
        gray_img = Mat(img.rows, img.cols, CV_8UC1);

        rows_per_process = img.rows / size;
        remainder_rows = img.rows % size;

        printf("Image loaded: %dx%d\n", img_rows, img_cols);
        printf("Distributing work: %d rows per process, %d extra rows\n", rows_per_process, remainder_rows);

        start_time = MPI_Wtime();  // Start timing
    }

    MPI_Bcast(&img_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Mat local_img(rows_per_process + (rank < remainder_rows ? 1 : 0), img_cols, CV_8UC3);
    Mat local_gray(local_img.rows, img_cols, CV_8UC1);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int start_row = i * rows_per_process + std::min(i, remainder_rows);
            int end_row = start_row + rows_per_process + (i < remainder_rows ? 1 : 0);

            if (i == 0) {
                img(Rect(0, start_row, img_cols, end_row - start_row)).copyTo(local_img);
            } else {
                MPI_Send(img.ptr(start_row), (end_row - start_row) * img_cols * 3, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_img.data, local_img.rows * img_cols * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Process %d adjusting brightness and contrast, converting rows %d to %d\n", rank, rank * rows_per_process, rank * rows_per_process + local_gray.rows);

    adjust_brightness(local_img, brightness);  // Adjust brightness
    adjust_contrast(local_img, contrast);      // Adjust contrast
    convert_to_grayscale(local_img, local_gray, 0, local_gray.rows);

    printf("Process %d completed conversion.\n", rank);

    if (rank == 0) {
        local_gray.copyTo(gray_img(Rect(0, 0, img_cols, local_gray.rows)));

        for (int i = 1; i < size; ++i) {
            int start_row = i * rows_per_process + std::min(i, remainder_rows);
            int end_row = start_row + rows_per_process + (i < remainder_rows ? 1 : 0);

            MPI_Recv(gray_img.ptr(start_row), (end_row - start_row) * img_cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        imwrite(output_image, gray_img);
        printf("Image processing complete. Output saved as %s\n", output_image);

        end_time = MPI_Wtime();  // End timing
        printf("Total processing time: %f seconds\n", end_time - start_time);

        int histogram[256];
        calculate_histogram(gray_img, histogram);
        printf("Histogram of the grayscale image:\n");
        print_histogram(histogram);
    } else {
        MPI_Send(local_gray.data, local_gray.rows * img_cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    printf("Process %d finished and exiting.\n", rank);

    MPI_Finalize();
    return 0;
}
