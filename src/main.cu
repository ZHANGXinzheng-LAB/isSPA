#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <filesystem>


#include "DataReader2.h"
#include "star_parser.h"
#include "helper.cuh"
#include "nonorm.cuh"
#include "norm.cuh"
#include "utils.h"

int main(int argc, char * argv[]) 
{
    try 
    {
        Config conf(argv[1]);
        conf.print(); // 逐行展示从配置文件中读取的各个参数

        const std::string filename = conf.gets("FSC");
        if (filename != "")
        {
            const std::vector<std::string> target_columns = {
            "_rlnResolution",
            "_rlnFourierShellCorrelationCorrected"
            };
            std::vector<float> k, fsc;
            //std::map<int, float> fsc;

            try {
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
            } catch (const std::exception& e) {
                std::cerr << "错误: " << e.what() << std::endl;
                return 1;
            }

            auto lst = LST::load(conf.gets("Input")); // 从Input中读取参数（欠焦值、像散等）
            EulerData euler(conf.gets("Euler_angles_file")); // 读取欧拉角

            auto device = conf.geti("GPU_ID");
            std::printf("\nSelected device ID: %d\n", device);

            auto first = std::max(0, std::min(conf.geti("First_image"), int(lst.size() - 1)));
            auto last = std::min(conf.geti("Last_image"), std::max(0, int(lst.size())));

            INIT_TIMEIT();

            Templates temp;
            std::printf("Picking templates: %s, ", conf.gets("Picking_templates").c_str());
            TIMEIT(temp = Templates(conf.gets("Picking_templates"), euler.size()));
            std::string output = conf.gets("Output");
            std::filesystem::path filePath = output;

            // 覆盖之前的文件
            std::fstream output1(output, std::ios::out | std::ios::trunc);

            if (filePath.extension() == ".star")
            {
                int bin = conf.geti("Bin");
                float pix_size = conf.getf("Pixel_size");
                float org_pix_size = pix_size / bin;
                std::fstream out_star(output, std::ios::out|std::ios::trunc);
                out_star << "\n# version 30001\n\ndata_optics\n\nloop_ \n_rlnOpticsGroupName #1 \n_rlnOpticsGroup #2 \n_rlnImageSize #3 \n_rlnMicrographOriginalPixelSize #4 \n_rlnVoltage #5 \n_rlnSphericalAberration #6 \n_rlnAmplitudeContrast #7 \n_rlnImagePixelSize #8 \n_rlnImageDimensionality #9 \n_rlnCtfDataAreCtfPremultiplied #10 \nopticsGroup1 1 " << conf.getf("Diameter") << " " << org_pix_size << " " << conf.getf("Voltage") << " " << conf.getf("Cs") << " " << conf.getf("Amplitude_contrast") << " " << pix_size << " 2 0 \n\n\n# version 30001\n\ndata_particles\n\nloop_ \n_rlnMicrographName #1 \n_rlnCoordinateX #2 \n_rlnCoordinateY #3 \n_rlnDefocusU #4 \n_rlnDefocusV #5 \n_rlnDefocusAngle #6 \n_rlnAngleRot #7 \n_rlnAngleTilt #8 \n_rlnAnglePsi #9 \n_rlnOpticsGroup #10 \n# isSPA score " <<  std::endl;
            }
            
            if (device != -1) 
            {
                for (auto i = first; i < last; ++i) 
                {
                    const auto & entry = lst[i];
                    if (conf.geti("Norm_type")) 
                    {
                        auto image = Image{entry};
                        auto params = image.p;
                        SearchNorm p(conf, euler, {params.width, params.height}, device);

                        TIMEIT(p.work_verbose(temp, image, output); std::printf("Device %d finished in ", device););
                    } 
                    /*
                    else 
                    {
                      SearchNoNorm p(conf, euler, {tile_size, tile_size}, device);
                      auto tiles = TileImages{entry};
                      TIMEIT(p.work_verbose(temp, tiles, output); std::printf("Device %d finished in ", device););
                    }
                    */
                }
            }
        }
        else
        {    
            auto lst = LST::load(conf.gets("Input")); // 从Input中读取参数（欠焦值、像散等）
            EulerData euler(conf.gets("Euler_angles_file")); // 读取欧拉角

            auto device = conf.geti("GPU_ID");
            std::printf("\nSelected device ID: %d\n", device);

            auto first = std::max(0, std::min(conf.geti("First_image"), int(lst.size() - 1)));
            auto last = std::min(conf.geti("Last_image"), std::max(0, int(lst.size())));

            INIT_TIMEIT();

            Templates temp;
            std::printf("Picking templates: %s, ", conf.gets("Picking_templates").c_str());
            TIMEIT(temp = Templates(conf.gets("Picking_templates"), euler.size()));
            std::string output = conf.gets("Output");
            std::filesystem::path filePath = output;

            // 覆盖之前的文件
            std::fstream output1(output, std::ios::out | std::ios::trunc);

            if (filePath.extension() == ".star")
            {
                int bin = conf.geti("Bin");
                float pix_size = conf.getf("Pixel_size");
                float org_pix_size = pix_size / bin;
                std::fstream out_star(output, std::ios::out|std::ios::trunc);
                out_star << "\n# version 30001\n\ndata_optics\n\nloop_ \n_rlnOpticsGroupName #1 \n_rlnOpticsGroup #2 \n_rlnImageSize #3 \n_rlnMicrographOriginalPixelSize #4 \n_rlnVoltage #5 \n_rlnSphericalAberration #6 \n_rlnAmplitudeContrast #7 \n_rlnImagePixelSize #8 \n_rlnImageDimensionality #9 \n_rlnCtfDataAreCtfPremultiplied #10 \nopticsGroup1 1 " << conf.getf("Diameter") << " " << org_pix_size << " " << conf.getf("Voltage") << " " << conf.getf("Cs") << " " << conf.getf("Amplitude_contrast") << " " << pix_size << " 2 0 \n\n\n# version 30001\n\ndata_particles\n\nloop_ \n_rlnMicrographName #1 \n_rlnCoordinateX #2 \n_rlnCoordinateY #3 \n_rlnDefocusU #4 \n_rlnDefocusV #5 \n_rlnDefocusAngle #6 \n_rlnAngleRot #7 \n_rlnAngleTilt #8 \n_rlnAnglePsi #9 \n_rlnOpticsGroup #10 \n# isSPA score " <<  std::endl;
            }
            
            if (device != -1) 
            {
                for (auto i = first; i < last; ++i) 
                {
                    const auto & entry = lst[i];
                    if (conf.geti("Norm_type")) 
                    {
                        auto image = Image{entry};
                        auto params = image.p;
                        SearchNorm p(conf, euler, {params.width, params.height}, device);

                        TIMEIT(p.work_verbose(temp, image, output); std::printf("Device %d finished in ", device););
                    } 
                    /*
                    else 
                    {
                      SearchNoNorm p(conf, euler, {tile_size, tile_size}, device);
                      auto tiles = TileImages{entry};
                      TIMEIT(p.work_verbose(temp, tiles, output); std::printf("Device %d finished in ", device););
                    }
                    */
                }
            }
        }
        /* 
        else 
        {
            auto devcount = GetDeviceCount();
            std::printf("Device count: %d\n", devcount);
            auto intervals = work_intervals(first, last, devcount);

            std::vector<std::stringstream> ss(devcount);

            auto worker = [&](int device, std::pair<int, int> interval) 
            {
                INIT_TIMEIT();
                std::stringstream output;
                for (auto i = interval.first; i < interval.second; ++i) 
                {
                    const auto& entry = lst[i];
                    if (conf.geti("Norm_type")) 
                    {
                        auto image = Image{entry};
                        auto params = image.p;
                        SearchNorm p(conf, euler, {params.width, params.height}, device);

                        p.work(temp, image, output);
                    } 
                    else 
                    {
                        SearchNoNorm p(conf, euler, {tile_size, tile_size}, device);
                        auto tiles = TileImages{entry};
                        TIMEIT(p.work(temp, tiles, output); std::printf("Device %d finished in ", device););
                    }
                }
                ss[device] = std::move(output);
            };

            auto wcount = std::min(devcount, last - first);
            std::vector<std::thread> ts(wcount);
            for (auto dev = 0; dev < wcount; ++dev) 
            {
                ts[dev] = std::thread(worker, dev, intervals[dev]);
            }

            for (auto& t : ts) 
            {
                t.join();
            }

            for (const auto& s : ss) 
            {
                output << s.rdbuf();
            }
        }
        */
    } 
    catch (const std::exception & e) 
    {
        std::cout << e.what() << std::endl;
        std::exit(-1);
    }
    
    return 0;
}