#include <fmt/color.h>
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "cv-helpers.hpp"
#include "devices/serial/serial.hpp"
#include "devices/catch_keyboard.hpp"
#include "OpenVINO/detector.h"
#include "solvePnP/solvePnP.hpp"
#include "utils.hpp"
#include "utils/mjpeg_streamer.hpp"

using namespace std::chrono_literals;

void topCameraThread(RoboInf &robo_inf,
    const std::shared_ptr<RoboSerial> &serial,
    const std::shared_ptr<nadjieb::MJPEGStreamer> &streamer_ptr) {
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30.f);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30.f);
  pipe.start(cfg);
  rs2::align align_to(RS2_STREAM_COLOR);

  Detector *detect_cube_top = new Detector;
  detect_cube_top->init(fmt::format("{}{}", SOURCE_PATH,
                        "/model/best.xml"), 0.5, 0.5);

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/d435i.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  cv::Mat src_img;
  cv::Rect object_3d_rect(0, 0, 140, 140);
  cv::Point2f pnp_angle;

  constexpr float errors_angle = 10.0f;
  constexpr float errors_distance = 8.0f;
  constexpr int duration_times = 3;
  constexpr int uart_sleep_t = 20000;
  while (true) try {
    usleep(1);
    static int cube_middle_detect_times{0};
    static int cude_front_detect_times{0};

    // 获取图像
    auto frames = pipe.wait_for_frames();
    auto depth_frame = frames.get_depth_frame();
    auto aligned_set = align_to.process(frames);
    auto color_frame = aligned_set.get_color_frame();
    src_img = frame_to_mat(color_frame);
    // 1280 * 720
    // fmt::print("the size of the image :{},{}",src_img.size().width,src_img.size().height);
    cv::line(src_img, cv::Point(src_img.cols * 0.25, 0),
              cv::Point(src_img.cols * 0.25, src_img.rows),
              cv::Scalar(0, 150, 255), 5);
    cv::line(src_img, cv::Point(src_img.cols *0.75, 0),
              cv::Point(src_img.cols *0.75, src_img.rows),
              cv::Scalar(0, 150, 255), 5);
    // 检测目标
    std::vector<Detector::Object> detected_objects;
    // cv::imshow("RGB",src_img);
    if(!detect_cube_top->process_frame(src_img, detected_objects)) {
      cv::cvtColor(src_img,src_img,cv::COLOR_RGB2BGR);
      // cv::resize(src_img,src_img,cv::Size(src_img.cols / cols_rate,src_img.rows / row_rate));
      cv::imshow("src",src_img);
      cv::waitKey(1);
#ifndef RELEASE
      if (!src_img.empty()) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", src_img, buff_bgr);
        streamer_ptr->publish("/tpcm",
                              std::string(buff_bgr.begin(), buff_bgr.end()));
      }
#endif
    continue;
    }

    std::sort(detected_objects.begin(), detected_objects.end());
    cv::Rect rect_sort = detected_objects.at(0).rect;
    // cv::resize(src_img,src_img,cv::Size(src_img.cols / cols_rate,src_img.rows / row_rate));
    cv::rectangle(src_img, rect_sort,
                    cv::Scalar(0, 255, 0), 2);
    // Detector::Object seleted_obj; // 夹取目标
    // 选择夹取目标
      switch (robo_inf.catch_cube_mode_status.load()) {
        // 旋转对位
        case CatchMode::spin: {
          cv::putText(src_img,
                      "spin mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          // pnp->solvePnP(object_3d_rect, seleted_obj.rect, pnp_angle,
          //               pnp_coordinate_mm, pnp_depth);
          pnp_angle.y = src_img.cols - (rect_sort.x + rect_sort.width * 0.5) - 300;
          if (cube_middle_detect_times < duration_times) {
            if (abs(pnp_angle.y) < errors_angle ) {
              cube_middle_detect_times++;
            } else {
              RoboSpinCmdUartBuff uart_temp_spin_cmd;
              uart_temp_spin_cmd.yaw_angle = pnp_angle.y;
              std::cout << "uart_temp_spin_cmd.yaw_angle" << uart_temp_spin_cmd.yaw_angle << "\n";
              serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
            }
          } else {
            RoboSpinCmdUartBuff uart_temp_spin_cmd;
            uart_temp_spin_cmd.yaw_angle = 0.f;
            for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_spin_cmd, sizeof(uart_temp_spin_cmd));
              std::cout << "send yaw 0\n";
              usleep(uart_sleep_t);
            }
            cube_middle_detect_times = 0;
            robo_inf.catch_cube_mode_status.store(CatchMode::go);
          }
          break;
        }

        // 距离对位
        case CatchMode::go: {
          cv::putText(src_img,
                      "go straight mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          // 通过方框在图像中位置判断
          float select_cube_dis = abs(src_img.rows - (rect_sort.y + rect_sort.height * 0.5) - 240);
          fmt::print("distance :[{}]\n",select_cube_dis);
          if (cude_front_detect_times < duration_times) {
            if (abs(select_cube_dis) < errors_distance) {
              cude_front_detect_times++;
            } else {
              RoboGoCmdUartBuff uart_temp_go_cmd;
              uart_temp_go_cmd.distance = select_cube_dis;
              std::cout << "uart_temp_go_cmd.distance:" << uart_temp_go_cmd.distance << "\n";
              serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
            }
          } else {
            RoboGoCmdUartBuff uart_temp_go_cmd;
            uart_temp_go_cmd.distance = 0.f;
            for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_go_cmd, sizeof(uart_temp_go_cmd));
              usleep(uart_sleep_t);
            }
            robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
            cude_front_detect_times = 0;
          }
          break;
        }

        case CatchMode::catch_cube: {
          cv::putText(src_img,
                      "catch cube mode",
                      cv::Point(50, 50),
                      cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
          RoboCatchCmdUartBuff uart_temp_catch_cmd;
          // 发送积木状态
          if(detected_objects.at(0).id == 0) {
            uart_temp_catch_cmd.cube_state = CUBE_LIE;
          }
          else
          {
            uart_temp_catch_cmd.cube_state = CUBE_ERECT;
          }
          for (int i = 0; i < 3; i++) {
              serial->write((uint8_t *)&uart_temp_catch_cmd, sizeof(uart_temp_catch_cmd));
              // std::cout << "catch, rect size:" << seleted_obj.rect.area() 
              // << "rect state:" << (int)uart_temp_catch_cmd.cube_state << "\n";
              usleep(uart_sleep_t);
          }
          robo_inf.catch_cube_mode_status.store(CatchMode::off);
          break;
        }

        case CatchMode::off:
          cube_middle_detect_times = 0;
          cude_front_detect_times = 0;
          break;
        case CatchMode::detect_mode:
        {
          RoboCatchCmdUartBuff uart_temp_catch_cmd;
          // 发送积木状态
          if(detected_objects.at(0).id == 0) {
            uart_temp_catch_cmd.cube_state = CUBE_LIE;
          }
          else
          {
            uart_temp_catch_cmd.cube_state = CUBE_ERECT;
          }
          for (int i = 0; i < 3; i++) {
            serial->write((uint8_t *)&uart_temp_catch_cmd, sizeof(uart_temp_catch_cmd));
                // std::cout << "catch, rect size:" << seleted_obj.rect.area() 
                // << "rect state:" << (int)uart_temp_catch_cmd.cube_state << "\n";
            usleep(uart_sleep_t);
          }
        }
        default:
          break;
      }

    cv::imshow("src",src_img);
    cv::waitKey(1);
#ifndef RELEASE
      if (!src_img.empty()) {
        std::vector<uchar> buff_bgr;
        cv::imencode(".jpg", src_img, buff_bgr);
        streamer_ptr->publish("/tpcm",
                              std::string(buff_bgr.begin(), buff_bgr.end()));
      }
#endif
    } catch (const std::exception &e) {
      fmt::print("{}\n", e.what());
    }
}

void uartReadThread(const std::shared_ptr<RoboSerial> &serial,
                    RoboInf &robo_inf) {
  while (true) try {
      if(serial->isOpen()) {
        serial->ReceiveInfo(robo_inf);
      } else {
        serial->open();
      }
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      serial->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {}\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void KeyboardThread(std::unique_ptr<CatchKeyboard> &kb, RoboInf &robo_inf) {
  unsigned short code;
  while (true) try {
    if (kb->GetEvent(code)) {
      switch (code) {
        case KEY_KP1:
          robo_inf.catch_cube_mode_status.store(CatchMode::spin);
          break;
        case KEY_KP2:
          robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
          break;
        case KEY_KP3:
          robo_inf.catch_cube_mode_status.store(CatchMode::off);
          break;
        default:
          break;
      }
    }
  } catch(const std::exception& e) {
    fmt::print("{}\n", e.what());
  }
}
int main(int argc, char *argv[]) {
  RoboInf robo_inf;

  auto streamer_ptr = std::make_shared<nadjieb::MJPEGStreamer>();
  streamer_ptr->start(8080);

  auto serial = std::make_shared<RoboSerial>("/dev/robo_tty", 115200);

  auto kb = std::make_unique<CatchKeyboard>("2.4G Wireless Keyboard");

  std::thread uart_thread(uartReadThread, std::ref(serial), std::ref(robo_inf));
  uart_thread.detach();

  std::thread top_camera_thread(topCameraThread, std::ref(robo_inf),
                                std::ref(serial), std::ref(streamer_ptr));
  top_camera_thread.detach();

  std::thread kb_thread(KeyboardThread, std::ref(kb), std::ref(robo_inf));
  kb_thread.detach();
  if (std::cin.get() == 'q') {
    top_camera_thread.~thread();
    uart_thread.~thread();
    kb_thread.~thread();
  }
  return 0;
}