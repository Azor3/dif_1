#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>

#define IMAGE_SIZE 28
#define RESIZE_DIM 24
#define VECTOR_SIZE ((RESIZE_DIM * RESIZE_DIM) + 1)
#define MAX_PATH 256
#define MAX_IMAGES 1000
#define TRAIN_SPLIT 0.8
#define LEARNING_RATE 0.01
#define EPOCHS 50
#define NUM_INIT 5

/* Adam hyperparameters */
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

typedef struct {
    double* data;
    int label;
} Sample;

typedef struct {
    Sample* samples;
    int count;
} Dataset;

/* Save weights to file */
void save_weights(double** weight_history, int epochs, int size, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return;
    }

    int i, j;
    for (i = 0; i < epochs; i++) {
        for (j = 0; j < size - 1; j++) {
            fprintf(fp, "%.6f,", weight_history[i][j]);
        }
        fprintf(fp, "%.6f\n", weight_history[i][size - 1]);
    }
    fclose(fp);
}

/* Initialize weights with fixed value */
void init_weights(double* w, int size, double init_value) {
    int i;
    for (i = 0; i < size; i++) {
        w[i] = init_value;
    }
}

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

double* load_and_process_image(const char* filename) {
    int width, height, channels;
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 1);
    if (!img) {
        fprintf(stderr, "Error loading image %s\n", filename);
        return NULL;
    }

    double* data = (double*)malloc(VECTOR_SIZE * sizeof(double));
    if (!data) {
        stbi_image_free(img);
        return NULL;
    }

    int i;
    for (i = 0; i < RESIZE_DIM * RESIZE_DIM; i++) {
        int orig_i = (i / RESIZE_DIM) * IMAGE_SIZE / RESIZE_DIM;
        int orig_j = (i % RESIZE_DIM) * IMAGE_SIZE / RESIZE_DIM;
        data[i] = img[orig_i * IMAGE_SIZE + orig_j] / 255.0;
    }
    data[VECTOR_SIZE - 1] = 1.0; /* Bias term */

    stbi_image_free(img);
    return data;
}

double tanh_custom(double x) {
    double exp_2x = exp(2 * x);
    return (exp_2x - 1) / (exp_2x + 1);
}

double forward(const double* w, const double* x, int size) {
    double sum = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        sum += w[i] * x[i];
    }
    return tanh_custom(sum);
}

double compute_loss(double pred, double true_val) {
    double diff = pred - true_val;
    return 0.5 * diff * diff;
}

void compute_gradient(double* grad, const double* w, const double* x, 
                     double pred, double true_val, int size) {
    double error = pred - true_val;
    double dtanh = 1.0 - pred * pred;
    int i;
    for (i = 0; i < size; i++) {
        grad[i] = error * dtanh * x[i];
    }
}

void shuffle_dataset(Dataset* dataset) {
    int i;
    for (i = dataset->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Sample temp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = temp;
    }
}

void process_directory(const char* dir_path, Dataset* dataset, int label, int* count) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", dir_path);
        exit(1);
    }

    struct dirent* entry;
    char full_path[MAX_PATH];
    while ((entry = readdir(dir)) != NULL && *count < MAX_IMAGES) {
        if (entry->d_name[0] == '.') continue;
        
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        double* data = load_and_process_image(full_path);
        if (data) {
            dataset->samples[*count].data = data;
            dataset->samples[*count].label = label;
            (*count)++;
        }
    }
    closedir(dir);
}

void gd(Dataset* train, double* w, int size, double lr, int epochs,
              const char* log_file, double** weight_history) {
    FILE* fp = fopen(log_file, "w");
    double* gradient = (double*)malloc(size * sizeof(double));
    double start_time = get_time();
    int i, j, epoch;

    for (epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (i = 0; i < size; i++) {
            gradient[i] = 0.0;
        }

        for (i = 0; i < train->count; i++) {
            double pred = forward(w, train->samples[i].data, size);
            double true_val = (double)train->samples[i].label;
            total_loss += compute_loss(pred, true_val);
            
            double* temp_grad = (double*)malloc(size * sizeof(double));
            compute_gradient(temp_grad, w, train->samples[i].data, pred, true_val, size);
            
            for (j = 0; j < size; j++) {
                gradient[j] += temp_grad[j];
            }
            free(temp_grad);
        }

        for (i = 0; i < size; i++) {
            gradient[i] /= train->count;
            w[i] -= lr * gradient[i];
            weight_history[epoch][i] = w[i];
        }

        fprintf(fp, "GD,%d,%.6f,%.6f\n", epoch, total_loss/train->count, get_time() - start_time);
    }

    free(gradient);
    fclose(fp);
}

void sgd(Dataset* train, double* w, int size, double lr, int epochs,
               const char* log_file, double** weight_history) {
    FILE* fp = fopen(log_file, "w");
    double* gradient = (double*)malloc(size * sizeof(double));
    double start_time = get_time();
    int i, j, epoch;

    for (epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        shuffle_dataset(train);

        for (i = 0; i < train->count; i++) {
            double pred = forward(w, train->samples[i].data, size);
            double true_val = (double)train->samples[i].label;
            total_loss += compute_loss(pred, true_val);

            compute_gradient(gradient, w, train->samples[i].data, pred, true_val, size);

            for (j = 0; j < size; j++) {
                w[j] -= lr * gradient[j];
            }
        }

        for (i = 0; i < size; i++) {
            weight_history[epoch][i] = w[i];
        }

        fprintf(fp, "SGD,%d,%.6f,%.6f\n", epoch, total_loss/train->count, get_time() - start_time);
    }

    free(gradient);
    fclose(fp);
}

void adam(Dataset* train, double* w, int size, double lr, int epochs,
                const char* log_file, double** weight_history) {
    FILE* fp = fopen(log_file, "w");
    double* m = (double*)calloc(size, sizeof(double));
    double* v = (double*)calloc(size, sizeof(double));
    double* gradient = (double*)malloc(size * sizeof(double));
    double start_time = get_time();
    int i, j, epoch;
    double t = 1.0;

    for (epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        shuffle_dataset(train);

        /* Reset gradient */
        for (i = 0; i < size; i++) {
            gradient[i] = 0.0;
        }

        /* Compute total gradient and loss */
        for (i = 0; i < train->count; i++) {
            double pred = forward(w, train->samples[i].data, size);
            double true_val = (double)train->samples[i].label;
            total_loss += compute_loss(pred, true_val);
            
            double* temp_grad = (double*)malloc(size * sizeof(double));
            compute_gradient(temp_grad, w, train->samples[i].data, pred, true_val, size);
            
            for (j = 0; j < size; j++) {
                gradient[j] += temp_grad[j];
            }
            free(temp_grad);
        }

        /* Average gradients */
        for (i = 0; i < size; i++) {
            gradient[i] /= train->count;
        }

        /* Adam update */
        for (i = 0; i < size; i++) {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * gradient[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * gradient[i] * gradient[i];
            
            double m_hat = m[i] / (1.0 - pow(BETA1, t));
            double v_hat = v[i] / (1.0 - pow(BETA2, t));
            
            w[i] -= lr * m_hat / (sqrt(v_hat) + EPSILON);
        }
        t += 1.0;

        /* Save weights and log results */
        for (i = 0; i < size; i++) {
            weight_history[epoch][i] = w[i];
        }

        total_loss /= train->count;
        fprintf(fp, "Adam,%d,%.6f,%.6f\n", epoch, total_loss, get_time() - start_time);
    }

    free(m);
    free(v);
    free(gradient);
    fclose(fp);
}

int main(void) {
    Dataset dataset;
    int total_count = 0;
    double* w;
    double** weight_history;
    int i;
    
    /* Define initial weight values */
    double init_values[5] = {0.0, 0.1, 0.01, 0.001, 0.0001};

    dataset.samples = (Sample*)malloc(MAX_IMAGES * 2 * sizeof(Sample));
    dataset.count = 0;

    printf("Loading images from mnist_jpg/0...\n");
    process_directory("mnist_jpg/0", &dataset, 1, &total_count);
    printf("Loading images from mnist_jpg/1...\n");
    process_directory("mnist_jpg/1", &dataset, -1, &total_count);
    dataset.count = total_count;

    printf("Loaded %d images total\n", dataset.count);

    int train_size = (int)(dataset.count * TRAIN_SPLIT);
    Dataset train_set;
    train_set.samples = (Sample*)malloc(train_size * sizeof(Sample));
    train_set.count = train_size;

    shuffle_dataset(&dataset);
    for (i = 0; i < train_size; i++) {
        train_set.samples[i] = dataset.samples[i];
    }

    w = (double*)malloc(VECTOR_SIZE * sizeof(double));
    weight_history = (double**)malloc(EPOCHS * sizeof(double*));
    for (i = 0; i < EPOCHS; i++) {
        weight_history[i] = (double*)malloc(VECTOR_SIZE * sizeof(double));
    }

    printf("Starting training with %d different initializations...\n", NUM_INIT);
    
    for (i = 0; i < NUM_INIT; i++) {
        printf("Running initialization %d/%d with w=%.4f\n", i + 1, NUM_INIT, init_values[i]);
        char log_file[MAX_PATH];
        char weight_file[MAX_PATH];
        
        /* Train with GD */
        init_weights(w, VECTOR_SIZE, init_values[i]);
        snprintf(log_file, sizeof(log_file), "gd_results_%d.csv", i);
        snprintf(weight_file, sizeof(weight_file), "weights_gd_%d.csv", i);
        gd(&train_set, w, VECTOR_SIZE, LEARNING_RATE, EPOCHS, log_file, weight_history);
        save_weights(weight_history, EPOCHS, VECTOR_SIZE, weight_file);
        
        /* Train with SGD */
        init_weights(w, VECTOR_SIZE, init_values[i]);
        snprintf(log_file, sizeof(log_file), "sgd_results_%d.csv", i);
        snprintf(weight_file, sizeof(weight_file), "weights_sgd_%d.csv", i);
        sgd(&train_set, w, VECTOR_SIZE, LEARNING_RATE, EPOCHS, log_file, weight_history);
        save_weights(weight_history, EPOCHS, VECTOR_SIZE, weight_file);
        
        /* Train with Adam */
        init_weights(w, VECTOR_SIZE, init_values[i]);
        snprintf(log_file, sizeof(log_file), "adam_results_%d.csv", i);
        snprintf(weight_file, sizeof(weight_file), "weights_adam_%d.csv", i);
        adam(&train_set, w, VECTOR_SIZE, LEARNING_RATE, EPOCHS, log_file, weight_history);
        save_weights(weight_history, EPOCHS, VECTOR_SIZE, weight_file);
    }

    /* Cleanup */
    printf("Cleaning up...\n");
    for (i = 0; i < dataset.count; i++) {
        free(dataset.samples[i].data);
    }
    free(dataset.samples);
    free(train_set.samples);
    free(w);
    
    for (i = 0; i < EPOCHS; i++) {
        free(weight_history[i]);
    }
    free(weight_history);

    printf("Done!\n");
    return 0;
}