#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>

// 定义数据块结构：支持任意列
struct StarBlock {
    std::string name; // 块名（如 "data_general"）
    std::vector<std::string> columns; // 列名（如 ["rlnResolution", ...]）
    std::vector<std::unordered_map<std::string, std::string>> rows; // 数据行
};

// 核心解析函数：从STAR文件中提取指定列的数据
std::vector<StarBlock> parse_star_file(
    const std::string& filename, 
    const std::vector<std::string>& target_columns = {}
) {
    std::vector<StarBlock> blocks;
    StarBlock* current_block = nullptr;
    bool in_loop = false;
    std::vector<size_t> target_indices;

    std::ifstream file(filename);
    if (!file) throw std::runtime_error("无法打开文件: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        // 去除行首尾空白字符
        line.erase(line.find_last_not_of("\r\n\t ") + 1);
        if (line.empty() || line[0] == '#') continue;

        // 检测数据块开始
        if (line.substr(0, 5) == "data_") {
            blocks.emplace_back();
            current_block = &blocks.back();
            current_block->name = line.substr(5);
            in_loop = false;
            continue;
        }

        if (!current_block) continue;

        // 检测循环开始
        if (line == "loop_") {
            in_loop = true;
            current_block->columns.clear();
            target_indices.clear();
            continue;
        }

        // 解析列名
        if (in_loop && line[0] == '_') {
            std::istringstream iss(line);
            std::string col_name;
            iss >> col_name;
            current_block->columns.push_back(col_name);

            // 记录目标列的索引
            if (!target_columns.empty()) {
                for (size_t i = 0; i < target_columns.size(); ++i) {
                    if (col_name == target_columns[i]) {
                        target_indices.push_back(current_block->columns.size() - 1);
                    }
                }
            }
            continue;
        }

        // 解析数据行
        if (in_loop && !current_block->columns.empty()) {
            std::istringstream iss(line);
            std::vector<std::string> values;
            std::string value;

            while (iss >> value) {
                values.push_back(value);
            }

            // 提取目标列数据（或全部列）
            std::unordered_map<std::string, std::string> row;
            if (target_columns.empty()) {
                for (size_t i = 0; i < current_block->columns.size() && i < values.size(); ++i) {
                    row[current_block->columns[i]] = values[i];
                }
            } else {
                for (size_t idx : target_indices) {
                    if (idx < values.size()) {
                        row[current_block->columns[idx]] = values[idx];
                    }
                }
            }

            if (!row.empty()) {
                current_block->rows.push_back(row);
            }
        }
    }

    return blocks;
}