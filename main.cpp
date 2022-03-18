#include "devices/camera/mv_video_capture.hpp"
#include "infer/detector.h"
#include "omp.h"

int main(int argc, char const *argv[])
{

    int camera_exposure = 10000;
    mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_640_X_480,
                                          camera_exposure);
    mindvision::VideoCapture *mv_capture = new mindvision::VideoCapture(camera_params);
    // auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);
    cv::Mat src_img;
    string xml_path = "/home/wolf/Code_Pack/R2_SMS/model/best.xml";
    Detector *detector = new Detector(xml_path);
    detector->init( 0.6, 0.3);
    Mat draw_img;
    while (1)
    {
        if (mv_capture->isindustryimgInput())
        {
            mv_capture->cameraReleasebuff();
            src_img = mv_capture->image();
            auto start = std::chrono::high_resolution_clock::now();
            if (!src_img.empty()) {
                draw_img = src_img.clone();
                cv::resize(draw_img,draw_img,cv::Size(640,640));
                try
                {
                    vector<Detector::Object> detected_objects;

                    if (detector->process_frame(src_img, detected_objects)) {
                        for (int i = 0; i < detected_objects.size(); ++i) {
                            int xmin = detected_objects[i].rect.x;
                            int ymin = detected_objects[i].rect.y;
                            int width = detected_objects[i].rect.width;
                            int height = detected_objects[i].rect.height;
                            Rect rect(xmin, ymin, width, height); //左上坐标（x,y）和矩形的长(x)宽(y)
                            cv::rectangle(draw_img, rect, Scalar(200, 0, 200), 2, LINE_AA, 0);
                            cv::putText(draw_img, detected_objects[i].name + " " + to_string(detected_objects[i].status),
                                            cv::Point(rect.x, rect.y - 5), 1, 1, Scalar(200, 0, 255), 1, LINE_4);
                        }
                    }

                }
                catch(const std::exception& e) {
                    std::cerr << e.what() << '\n';
                }
                imshow("cap", src_img);
                imshow("draw",draw_img);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            fmt::print("use:{}s\n", 1.0/diff.count());
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }

    }
    detector->uninit();
}
