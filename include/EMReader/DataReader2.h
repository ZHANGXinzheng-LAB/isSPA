#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>
#include <iostream>
#include <filesystem>

using namespace std::string_literals; // makes visible operator""s

template <class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct Config 
{
    
    using Node = std::variant<int, float, std::string>; // 输入参数可能是整数、小数、字符串
    std::unordered_map<std::string, Node> value
    {
        // 参数列表
        {"Input", ""s},
        {"Picking_templates", ""s},
        {"Template_dimensions", 2},
        {"Euler_angles_file", ""s},
        {"Pixel_size", -1.0f},
        {"Phi_step", -1.0f},
        {"n", 1.0f},
        {"Voltage", 300.0f}, //单位 kV
        {"Cs", 2.7f}, // 单位 mm
        {"Amplitude_contrast", 0.1f},
        {"Highest_resolution", 8.0f}, //单位 Angstrom
        {"Lowest_resolution", 400.0f},
        {"Diameter", -1.0f},
        // optional
        {"Bin", 1.0f},
        {"Score_threshold", 6.4f},
        {"Output", ""s},
        {"First_image", 0},
        {"Last_image", 0},
        {"GPU_ID", 0},
        {"Window_size", 512},
        {"Phase_flip", 1},
        {"Overlap", 0},
        {"Norm_type", 1},
        {"Invert", 1},
        {"Fourier_padding", 0}, 
        {"FSC", ""s}, 
        {"n(k)", ""s}
    };

    std::string & gets(const std::string & key) { return std::get<std::string>(value[key]); }
    const std::string & gets(const std::string & key) const { return std::get<std::string>(value.at(key)); }
    int & geti(const std::string & key) { return std::get<int>(value[key]); }
    const int & geti(const std::string & key) const { return std::get<int>(value.at(key)); }
    float & getf(const std::string & key) { return std::get<float>(value[key]); }
    const float & getf(const std::string & key) const { return std::get<float>(value.at(key)); }

    void print() 
    {
        for (auto&& [name, val] : value) 
        {
          std::cout << name << " = ";
          std::visit(overloaded{
                         [](auto arg) { std::cout << arg << ' '; },
                         [](float arg) { std::cout << std::fixed << std::setprecision(3) << arg << ' '; },
                         [](const std::string& arg) { std::cout << arg << ' '; },
                     },
                     val);
          std::cout << std::endl;
        }
    }

    // Parse from config file
    Config(const std::string & path);

    // Check all required parameters
    void checkRequestPara();
};

struct Parameters 
{
    float apix{}, kk{}, energy{}, cs{}, highres{}, lowres{}, d_m{}, thresh{};
    float lambda{}, dfu{}, dfv{}, ds{}, defocus{}, dfdiff{}, dfang{};
    float edge_width{8.0f}, ampconst{0.07f};
    float a{-10.81f}, b{1.f}, b2{0.32f}, bfactor{-18.17f}, bfactor2{-15.22f}, bfactor3{1.72f};

    Parameters() = default;
    Parameters(const Config & c) : apix(c.getf("Pixel_size")), kk(c.getf("n")), energy(c.getf("Voltage")), cs(c.getf("Cs")), highres(c.getf("Highest_resolution")), lowres(c.getf("Lowest_resolution")), d_m(c.getf("Diameter")), thresh(c.getf("Score_threshold")) {}
};

struct EulerData 
{
    std::vector<float> euler1, euler2, euler3;

    EulerData() = default;
    EulerData(const std::string & eulerf);
    size_t size() const {return euler1.size();};
};

struct LST 
{
    struct Entry 
    {
        int unused; // 未使用的参数
        std::string rpath; // 文件路径
        double defocus; // 欠焦值
        double dfdiff; 
        double dfang;
    };

    static std::vector<Entry> load(const std::string & lst_path);
    static void print(const std::vector<Entry> & lst);
};

struct TextFileData
{
    std::vector<float> k, fsc, k1, n_k;
    
    TextFileData() = default;
    void read_fsc_star(const std::string & filename);
    void read_two_column_txt(const std::string & filename);
    void print_all();
};