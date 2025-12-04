#include <cassert>
#include <cstdio>

#include "DataReader2.h"
#include "star_parser.h"

// Parse config file
Config::Config(const std::string & path) 
{
    std::ifstream conf(path);
    if (!conf) 
    {
        std::printf("Configuration file does NOT exist: %s\n\n", path.c_str());
        return;
    }
    std::string key, e, val;
    // 逐行分空格读取字符串
    while (conf >> key >> e >> val) 
    {
        assert(e == "=");
        if (this->value.find(key) == this->value.end())
            continue;

        std::visit(overloaded{
                   [&](auto& arg) { arg = val; },
                   [&](int& arg) { arg = std::stoi(val); },
                   [&](float& arg) { arg = std::stof(val); },
                   [&](std::string& arg) { arg = val; },
               }, this->value[key]);
    }
    checkRequestPara();
}

void Config::checkRequestPara() 
{
    using namespace std;
    if (get<string>(value["Input"]).empty())
        printf("Error : lst/star file is required.\n");
    if (get<string>(value["Picking_templates"]).empty())
        printf("Error : File containing picking templates is required.\n");
    if (get<string>(value["Euler_angles_file"]).empty())
        printf("Error : File containing Euler angles is required.\n");
    if (get<float>(value["Pixel_size"]) < 0)
        printf("Error : Pixel size (Angstrom) is required.\n");
    //if (get<float>(value["Phi_step"]) < 0)
    //  printf("Error : Search step (degree) of angle phi is required.\n");
    if (get<float>(value["n"]) < 0)
        printf("Error : n is required.\n");
    if (get<float>(value["Voltage"]) < 0)
        printf("Error : Voltage (kV) is required.\n");
    if (get<float>(value["Cs"]) < 0)
        printf("Error : Spherical aberration (mm) is required.\n");
    if (get<float>(value["Highest_resolution"]) < 0)
        printf("Error : Highest resolution (Angstrom) is required.\n");
    if (get<float>(value["Lowest_resolution"]) < 0)
        printf("Error : Lowest resolution (Angstrom) is required.\n");
    if (get<float>(value["Diameter"]) < 0)
        printf("Error : Particle diameter (Angstrom) is required.\n");
}

EulerData::EulerData(const std::string & eulerf) 
{
    std::ifstream eulerfile(eulerf);
    if (!eulerfile) 
    {
        std::printf("File containing Euler angles does NOT exist: %s\n\n", eulerf.c_str());
        return;
    }
    // float x, y, z;
    // while (eulerfile >> x >> y >> z) {
    //   this->euler1.push_back(x);
    //   this->euler2.push_back(y);
    //   this->euler3.push_back(z);
    // }
    float alt, az, phi;
    std::string line;
    std::filesystem::path filePath = eulerf;
    if (!eulerfile.is_open())
        std::cout << "Failed to open " << eulerf << '\n';
    else
    {
        // 从角度文件中读取欧拉角
        if (filePath.extension() == ".star")
        {
            // STAR格式
            while (getline(eulerfile, line))
            {
                if (sscanf(line.c_str(), "%f %f %f", &az, &alt, &phi) == 3)
                {
                    this->euler1.push_back(alt);
                    this->euler2.push_back(az+90);
                    this->euler3.push_back(phi-90);
                }
            }
        }
        else if (filePath.extension() == ".lst" || ".txt")
        {
            // LST格式
            while (std::getline(eulerfile, line)) 
            {
                if (std::sscanf(line.c_str(), "%*f %f %f %f", &alt, &az, &phi) == 3 ||
                    std::sscanf(line.c_str(), "%f %f %f", &alt, &az, &phi) == 3 ||
                    std::sscanf(line.c_str(), "%*d %*s %f %f %f", &alt, &az, &phi) == 3)
                    // %*f跳过一个float
                {
                    this->euler1.push_back(alt);
                    this->euler2.push_back(az);
                    this->euler3.push_back(phi);
                }
            }
        }
    }
}

std::vector<LST::Entry> LST::load(const std::string & lst_path) 
{
    // 诸行读取lst文件中的参数，储存在用Entry构成的矢量中
    std::ifstream lstfile{lst_path};
    if (!lstfile) 
    {
        std::printf("LST file does NOT exist: %s\n\n", lst_path.c_str());
        return {};
    }

    std::vector<Entry> ret;
    std::string line;
    Entry e;
    while (std::getline(lstfile, line)) 
    {
        if (line.length()) 
        {
            if (line[0] == '#') 
                continue; // 跳过注释
        }
        char buf[1024] = {'\0'};
        std::sscanf(line.c_str(), "%d %1023s defocus=%lf dfdiff=%lf dfang=%lf", &e.unused, buf, &e.defocus, &e.dfdiff, &e.dfang); // 读取.lst文件中的参数，单位微米
        e.rpath = std::string{buf};
        ret.emplace_back(std::move(e)); // 将e中的数据剪切并粘贴到矢量ret中
    }
    return ret;
}

void LST::print(const std::vector<LST::Entry> & lst) 
{
    for (auto&& e : lst) 
    {
        std::printf("%d %s defocus=%lf dfdiff=%lf dfang=%lf\n", e.unused, e.rpath.c_str(), e.defocus, e.dfdiff, e.dfang);
    }
}

void TextFileData::read_two_column_txt(const std::string & filename) 
{
    std::ifstream file(filename);
    if (!file.is_open()) 
    {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) 
    {
        line_number++;
        // 跳过空行和注释行（以#开头）
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        float val1, val2;
        
        // 尝试读取两个双精度值
        if (!(iss >> val1 >> val2)) 
        {
            std::cerr << "警告: 第 " << line_number << " 行格式错误，已跳过\n";
            continue;
        }

        // 检查是否还有多余数据
        std::string remaining;
        if (iss >> remaining) 
        {
            std::cerr << "警告: 第 " << line_number << " 行包含多余数据，已忽略\n";
        }

        k1.push_back(val1);
        n_k.push_back(val2);
    }

    // 验证数据一致性
    if (k1.size() != n_k.size()) 
    {
        throw std::runtime_error("文件包含不一致的行数据");
    }
}

void TextFileData::read_fsc_star(const std::string & filename) 
{
    const std::vector<std::string> target_columns = 
    {
        "_rlnResolution",
        "_rlnFourierShellCorrelationCorrected"
    };

    auto blocks = parse_star_file(filename, target_columns);

    // 遍历所有块查找目标数据
    for (const auto& block : blocks) 
    {
        if (!block.rows.empty() && 
            block.columns.size() >= target_columns.size()) 
        {                
            for (size_t i = 0; i < block.rows.size(); ++i) 
            {
                const auto& row = block.rows[i];
                k.push_back(std::stof(row.at("_rlnResolution")));
                //fsc.push_back(std::stof(row.at("_rlnFourierShellCorrelationCorrected")));
                fsc.push_back(std::stof(row.at("_rlnFourierShellCorrelationCorrected")));
            }
            break;
        }
    }
    // 验证数据一致性
    if (k.size() != fsc.size()) 
    {
        throw std::runtime_error("文件包含不一致的行数据");
    }
}

void TextFileData::print_all()
{
    std::cout << "FSC(k) data:" << std::endl;
    for (size_t i = 0; i < k.size(); i++)
    {
        std::cout << k[i] << "\t" << fsc[i] << std::endl;
    }
    std::cout << "n(k) data:" << std::endl;
    for (size_t i = 0; i < k1.size(); i++)
    {
        std::cout << k1[i] << "\t" << n_k[i] << std::endl;
    }
}