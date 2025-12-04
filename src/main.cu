#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <filesystem>


#include "DataReader2.h"
#include "helper.cuh"
#include "nonorm.cuh"
#include "norm.cuh"
#include "utils.h"

int main(int argc, char * argv[]) 
{
    try 
    {
        int n_task = 1;
        Config conf(argv[1]);
        conf.print(); // 逐行展示从配置文件中读取的各个参数

        TextFileData text_data;
        std::string filename = conf.gets("FSC");
        if (filename != "")
        {
            text_data.read_fsc_star(filename);
        }
        filename = conf.gets("n(k)");
        if (filename != "")
        {
            text_data.read_two_column_txt(filename);
        }
        //text_data.print_all();

        auto lst = LST::load(conf.gets("Input")); // 从Input中读取参数（欠焦值、像散等）
        EulerData euler(conf.gets("Euler_angles_file")); // 读取欧拉角

        auto device = conf.geti("GPU_ID");
        std::printf("\nSelected device ID: %d\n", device);

        const int fourier_pad = conf.geti("Fourier_padding");

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

        auto & entry = lst[first];
        auto image = Image{entry};
        auto params = image.p;
        SearchNorm p(conf, euler, {params.width, params.height}, text_data, device, fourier_pad);
        //std::printf("Total number of euler sampling: %d", euler.size());
        std::printf("Device %d: Loading templates\n", device);
        p.LoadTemplate(temp);
        std::printf("Device %d: Preprocessing templates\n", device);
        p.PreprocessTemplate();

        //p.WriteTemplates("./preprocessed_templates.mrcs", 512, 512, 200, 1.189);

        if (device != -1) 
        {
            for (auto i = first; i < last; ++i) 
            {
                if (conf.geti("Norm_type")) 
                {
                    if (i == first)
                    {
                        TIMEIT(p.work_verbose(image, output); std::printf("Device %d finished in ", device););
                    }
                    else
                    {
                        entry = lst[i];
                        image = Image{entry};
                        TIMEIT(p.work_verbose(image, output); std::printf("Device %d finished in ", device););
                    }
                }
                printf("%d micrographs are finished\n", n_task); 
                n_task++;
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