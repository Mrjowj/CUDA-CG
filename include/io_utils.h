/**
 * @file io_utils.h
 * @brief Common Binary I/O Utilities
 */
#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

// 通用二进制写入
template <typename T>
void write_binary_file(const char* filename, T* data, size_t count) {
    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::perror("File opening failed");
        exit(1);
    }
    std::fwrite(data, sizeof(T), count, fp);
    std::fclose(fp);
    printf("[IO Write] Saved: %s (%zu elements)\n", filename, count);
}

// 通用二进制读取
template <typename T>
void read_binary_file(const char* filename, T* data, size_t count) {
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) {
        // 如果文件不存在，给出明确提示
        fprintf(stderr, "Error: Cannot open file '%s'. Did you run the generator first?\n", filename);
        exit(1);
    }
    size_t read_count = std::fread(data, sizeof(T), count, fp);
    std::fclose(fp);
    
    if (read_count != count) {
        fprintf(stderr, "Error: Expected %zu elements but read %zu from %s\n", count, read_count, filename);
        exit(1);
    }
    printf("[IO Read] Loaded: %s\n", filename);
}

#endif // IO_UTILS_H