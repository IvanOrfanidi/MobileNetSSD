#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include <boost/program_options.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

constexpr std::string_view NAME_LABEL_FILE = "labelmap.txt"; // Tag file
constexpr std::string_view NAME_DEPLOY_FILE = "MobileNetSSD_deploy.prototxt"; // Description file
constexpr std::string_view NAME_MODEL_FILE = "MobileNetSSD_deploy.caffemodel"; // Training files

constexpr int WIDTH = 500;
constexpr int HEIGHT = 500;
constexpr int DELAY_MS = 10;

void getLabelsFromFile(std::vector<std::string>& labels, const std::string& nameFile)
{
    std::ifstream file;
    file.open(nameFile, std::ifstream::in);
    if (file.is_open()) {
        while (!file.eof()) {
            std::string line;
            std::getline(file, line);
            std::stringstream stream(line);

            std::string name;
            stream >> name;
            labels.push_back(std::move(name));
        }
        file.close();
    }
}

int main(int argc, char* argv[])
{
    std::string inputFile;
    std::string outputFile;
    bool useCuda;
    boost::program_options::options_description desc("Options");
    desc.add_options()
        // All options:
        ("in,i", boost::program_options::value<std::string>(&inputFile)->default_value(""), "Path to input file.\n") //
        ("out,o", boost::program_options::value<std::string>(&outputFile)->default_value("output.mp4"), "Path to output file.\n") //
        ("cuda,c", boost::program_options::value<bool>(&useCuda)->default_value(true), "Set CUDA Enable.\n") //
        ("help,h", "Produce help message."); // Help
    boost::program_options::variables_map options;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), options);
        boost::program_options::notify(options);
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
    if (options.count("help")) {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }

    cv::VideoCapture capture;
    if (inputFile.length() == 0) {
        // Open default video camera
        capture.open(cv::VideoCaptureAPIs::CAP_ANY);
    } else {
        capture.open(inputFile);
    }
    if (!capture.isOpened()) {
        std::cerr << "Cannot open video!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string path = std::filesystem::current_path().string() + '/';
    std::replace(path.begin(), path.end(), '\\', '/');

    const auto width = capture.get(cv::CAP_PROP_FRAME_WIDTH); // Get width of frames of video
    const auto height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); // Get height of frames of video
    const auto fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Resolution of video: " << width << " x " << height << ".\nFrames per seconds: " << fps << "." << std::endl;

    std::vector<std::string> labels;
    getLabelsFromFile(labels, path + NAME_LABEL_FILE.data());
    if (labels.empty()) {
        std::cerr << "Failed to read file!" << std::endl;
        return EXIT_FAILURE;
    }

    // Define codec and create VideoWriter object.output is stored in 'outcpp.avi' file
    cv::VideoWriter video(outputFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(WIDTH, HEIGHT));

    bool cudaEnable = false;
    if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible() && useCuda) {
            cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
            cudaEnable = true;
        }
    }

    static constexpr int ESCAPE_KEY = 27;
    while (cv::waitKey(DELAY_MS) != ESCAPE_KEY) {
        // Read a new frame from video.
        cv::Mat source;
        if (capture.read(source) == false) {
            std::cerr << "Video camera is disconnected!" << std::endl;
            return EXIT_FAILURE;
        }
        resize(source, source, cv::Size(WIDTH, HEIGHT), 0, 0);

        cv::dnn::Net neuralNetwork;
        // Read binary file and description file
        neuralNetwork = cv::dnn::readNetFromCaffe(path + NAME_DEPLOY_FILE.data(), path + NAME_MODEL_FILE.data());
        if (neuralNetwork.empty()) {
            std::cerr << "Could not load Caffe_net!" << std::endl;
            return EXIT_FAILURE;
        }

        // Set CUDA as preferable backend and target
        if (cudaEnable) {
            neuralNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            neuralNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }

        const auto startTime = cv::getTickCount();
        static constexpr double SCALEFACTOR = 0.007843; // Is the only one determined in the author's documentation, this is the parameter
        const cv::Mat blob = cv::dnn::blobFromImage(source, // Input the image to be processed or classified by the neural network
            SCALEFACTOR, // After the image is subtracted from the average value, the remaining pixel values ​​are scaled to a certain extent
            cv::Size(300, 300), // Neural network requires the input image size during training.
            cv::Scalar(127.5, 127.5, 127.5) /* Mean needs to subtract the average value of the image as a whole.
                                      If we need to subtract different values ​​from the three channels of the RGB image,
                                      then 3 sets of average values ​​can be used. */
        );
        neuralNetwork.setInput(blob, "data");
        cv::Mat score = neuralNetwork.forward("detection_out");
        std::string runTime = "run time: " + std::to_string(static_cast<double>(cv::getTickCount() - startTime) / cv::getTickFrequency());
        runTime.erase(runTime.end() - 3, runTime.end());
        runTime += "s";

        const cv::Mat result(score.size[2], score.size[3], CV_32F, score.ptr<float>());

        static constexpr float CONFIDENCE_THRESHOLD = 0.3;
        for (int i = 0; i < result.rows; i++) {
            const float confidence = result.at<float>(i, 2);
            if (confidence > CONFIDENCE_THRESHOLD) {
                const size_t index = static_cast<size_t>(result.at<float>(i, 1));
                const float tl_x = result.at<float>(i, 3) * source.cols;
                const float tl_y = result.at<float>(i, 4) * source.rows;
                const float br_x = result.at<float>(i, 5) * source.cols;
                const float br_y = result.at<float>(i, 6) * source.rows;

                const cv::Rect objrect(static_cast<int>(tl_x), static_cast<int>(tl_y), static_cast<int>(br_x - tl_x), static_cast<int>(br_y - tl_y));
                cv::rectangle(source, objrect, cv::Scalar(0, 0, 255), 1, 8, 0);
                cv::putText(source, labels[index], cv::Point(tl_x, tl_y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 1, 5);
            }
        }

        cv::putText(source, runTime, cv::Point(10, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#ifdef NDEBUG
        cv::putText(source, "in release", cv::Point(180, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#else
        cv::putText(source, "in debug", cv::Point(180, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
#endif
        if (cudaEnable) {
            cv::putText(source, "using GPUs", cv::Point(300, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        } else {
            cv::putText(source, "using CPUs", cv::Point(300, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);
        }
        const std::string resolution = std::to_string(source.size().width) + "x" + std::to_string(source.size().height);
        cv::putText(source, resolution, cv::Point(source.size().width - 80, source.size().height - 10), cv::FONT_HERSHEY_PLAIN, 1.1, cv::Scalar(0, 255, 0), 1, 5);

        cv::imshow("MobileNet-demo", source);
        video.write(source);
    }

    capture.release();
    video.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
