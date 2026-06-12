#include <cassert>
#include <cstdio>

#include "DataReader2.h"

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
    if (get<float>(value["Phi_step"]) < 0)
        printf("Error : Search step (degree) of angle phi is required.\n");
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
        else if (filePath.extension() == ".lst")
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
