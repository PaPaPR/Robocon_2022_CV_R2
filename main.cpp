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

#include "TensorRTx/yolov5.hpp"
#include "cv-helpers.hpp"
#include "devices/serial/serial.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "solvePnP/solvePnP.hpp"
#include "utils.hpp"
#include "utils/mjpeg_streamer.hpp"

using namespace std::chrono_literals;

void topCameraThread(
    RoboCmd &robo_cmd, RoboInf &robo_inf,
    const std::shared_ptr<RoboSerial> &serial) {
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30.f);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30.f);
  pipe.start(cfg);
  rs2::align align_to(RS2_STREAM_COLOR);

  auto detect_cube_top = std::make_shared<YOLOv5TRT>(
      fmt::format("{}{}", SOURCE_PATH, "/models/Cube_top.engine"));

  auto pnp = std::make_shared<solvepnp::PnP>(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/d435i.xml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));

  cv::Mat src_img;
  cv::Rect object_2d_rect;
  cv::Rect object_3d_rect(0, 0, 140, 140);
  cv::Point2f pnp_angle;
  cv::Point3f pnp_coordinate_mm;
  float pnp_depth;
  int yolo_res_selected_id;

  while (cv::waitKey(1) != 'q') try {
    static int cube_middle_detect_times{0};
    static int cude_front_detect_times{0};
    if (robo_inf.catch_cube_mode.load()) {
      if (robo_inf.catch_cube_mode_status.load() == CatchMode::wait) {
        robo_inf.catch_cube_mode_status.store(CatchMode::spin);}
    } else if (robo_inf.detect_cube_mode.load()) {
      if (robo_inf.catch_cube_mode_status.load() == CatchMode::wait) {
        robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);}
    } else if (!robo_inf.catch_cube_mode.load() &&
               !robo_inf.detect_cube_mode.load()) {
      robo_inf.catch_cube_mode_status.store(CatchMode::wait);
      cube_middle_detect_times = 0;
      cude_front_detect_times = 0;
      continue;
    }

    auto frames = pipe.wait_for_frames();
    auto depth_frame = frames.get_depth_frame();
    auto aligned_set = align_to.process(frames);
    auto color_frame = aligned_set.get_color_frame();
    src_img = frame_to_mat(color_frame);
    auto res = detect_cube_top->Detect(src_img);

    //选择检测方框在视野中最下位置的块
    if (rectFilter(res, src_img, object_2d_rect, yolo_res_selected_id)) {
      switch (robo_inf.catch_cube_mode_status.load())
      {
      case CatchMode::spin: {
        pnp->solvePnP(object_3d_rect, object_2d_rect, pnp_angle,
                      pnp_coordinate_mm, pnp_depth);

        if (cube_middle_detect_times < 10) {
          if (pnp_angle.y < 1.5f && pnp_angle.y > -1.5f) {
            cube_middle_detect_times++;
          } else {
            RoboSpinCmdUartBuff uart_temp_struct;
            uart_temp_struct.yaw_angle = pnp_angle.y;
            serial->write((uint8_t *)&uart_temp_struct, sizeof(uart_temp_struct));
            cv::putText(src_img, "spin angle:" + std::to_string(uart_temp_struct.yaw_angle),
                        cv::Point(0, 100), cv::FONT_HERSHEY_DUPLEX, 1,
                        cv::Scalar(0, 150, 255), 1);
          }
        } else {
          RoboSpinCmdUartBuff uart_temp_struct;
          uart_temp_struct.yaw_angle = 0.f;
          for (int i = 0; i < 3; i++)
            serial->write((uint8_t *)&uart_temp_struct, sizeof(uart_temp_struct));

          cube_middle_detect_times = 0;
          robo_inf.catch_cube_mode_status.store(CatchMode::go);
        }
        break;
      }

      case CatchMode::go: {
        cv::Mat depth_frame_Mat = depth_frame_to_meters(depth_frame);
        cv::Mat depth_frame_Mat_mean_mask;
        cv::Rect object_2d_measure_depth_rect(object_2d_rect.x + object_2d_rect.width / 3,
                                              object_2d_rect.y + object_2d_rect.width * 0.1,
                                              object_2d_rect.width / 3,
                                              30);
        //去除距离为 0 的点
        cv::Mat object_2d_measure_depth_rect_roi = depth_frame_Mat(object_2d_measure_depth_rect);
        cv::inRange(object_2d_measure_depth_rect_roi, 0.01f, 2.f, depth_frame_Mat_mean_mask);
        cv::rectangle(src_img, object_2d_measure_depth_rect, cv::Scalar(0, 255, 0));
        cv::Scalar object_2d_measure_depth_rect_avg_dist = cv::mean(object_2d_measure_depth_rect_roi,
                                                                    depth_frame_Mat_mean_mask);

        cv::putText(src_img, std::to_string(object_2d_measure_depth_rect_avg_dist[0]),
                    cv::Point(0, 100), cv::FONT_HERSHEY_DUPLEX, 1,
                    cv::Scalar(0, 150, 255), 1);
        
        if (cude_front_detect_times > 10) {
          if (object_2d_measure_depth_rect_avg_dist[0] < 0.1) {
            cude_front_detect_times++;
          } else {
            RoboGoCmdUartBuff uart_temp_struct;
            uart_temp_struct.distance = object_2d_measure_depth_rect_avg_dist[0];
            serial->write((uint8_t *)&uart_temp_struct, sizeof(uart_temp_struct));
          }
        } else {
          RoboGoCmdUartBuff uart_temp_struct;
          uart_temp_struct.distance = 0.f;
          for (int i = 0; i < 3; i++)
            serial->write((uint8_t *)&uart_temp_struct, sizeof(uart_temp_struct));

          robo_inf.catch_cube_mode_status.store(CatchMode::catch_cube);
          cude_front_detect_times = 0;
        }
        break;
      }

      case CatchMode::catch_cube: {
        RoboCatchCmdUartBuff uart_temp_struct2;
        //To-do: 取多次识别的结果发送
        if ((int)res[yolo_res_selected_id].class_id == 0 ||
            (int)res[yolo_res_selected_id].class_id == 3) {
          uart_temp_struct2.cube_state = 0x01;
        } else if ((int)res[yolo_res_selected_id].class_id == 1 ||
                    (int)res[yolo_res_selected_id].class_id == 4) {
          uart_temp_struct2.cube_state = 0x02;
        } else if ((int)res[yolo_res_selected_id].class_id == 2 ||
                    (int)res[yolo_res_selected_id].class_id == 5) {
          uart_temp_struct2.cube_state = 0x03;
        }

        for (int i = 0; i < 3; i++)
          serial->write((uint8_t *)&uart_temp_struct2, sizeof(uart_temp_struct2));
        fmt::print("catch sign send.\n");

        robo_inf.catch_cube_mode_status.store(CatchMode::wait);
        break;
      }

      default:
        break;
      }

#ifndef RELEASE
      for (long unsigned int i = 0; i < res.size(); i++)
        cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
                      cv::Scalar(0, 255, 0), 2);
      cv::line(src_img, cv::Point(src_img.cols / 3, 0),
                cv::Point(src_img.cols / 3, src_img.rows),
                cv::Scalar(0, 150, 255), 2);
      cv::line(src_img, cv::Point(src_img.cols / 3 * 2, 0),
                cv::Point(src_img.cols / 3 * 2, src_img.rows),
                cv::Scalar(0, 150, 255), 2);

      cv::rectangle(src_img, object_2d_rect, cv::Scalar(0, 150, 255), 2);
      // 0-blue_yellow, 1-blue_white, 2-blue_blue, 3-red_yellow, 4-red_white,
      // 5-red_red
      cv::putText(src_img,
                  std::to_string((int)res[yolo_res_selected_id].class_id),
                  cv::Point(object_2d_rect.x, object_2d_rect.y - 1),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);
#endif
    }
#ifndef RELEASE
      cv::namedWindow("interface");
      cv::moveWindow("interface", 0, 0);
      if (!src_img.empty()) cv::imshow("interface", src_img);
#endif
      if (cv::waitKey(1) == 'q') break;
    } catch (const std::exception &e) {
      fmt::print("{}\n", e.what());
    }
}

void sideCameraThread(
    RoboCmd &robo_cmd, RoboInf &robo_inf,
    const std::shared_ptr<RoboSerial> &serial,
    const std::shared_ptr<nadjieb::MJPEGStreamer> &streamer_ptr) {
  int camera_exposure = 5000;
  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_1024,
                                        camera_exposure);
  auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);

  auto detect_cube_side = std::make_shared<YOLOv5TRT>(
      fmt::format("{}{}", SOURCE_PATH, "/models/Cube_side.engine"));

  cv::Mat src_img;
  cv::Rect object_2d_rect;
  int yolo_res_selected_id;

  while (true) try {
      if (mv_capture->isindustryimgInput() &&
          robo_inf.detect_cube_mode.load()) {
        mv_capture->cameraReleasebuff();
        src_img = mv_capture->image();

        auto res = detect_cube_side->Detect(src_img);

        if (sideRectFilter(res, src_img, object_2d_rect,
                           yolo_res_selected_id)) {
          //To-do: 拍摄侧边摄像头，块立着被夹子夹住的数据集
          if ((int)res[yolo_res_selected_id].class_id) {
            /* code */
          }
        }
        

#ifndef RELEASE
      for (long unsigned int i = 0; i < res.size(); i++)
        cv::rectangle(src_img, get_rect(src_img, res[i].bbox),
                      cv::Scalar(0, 255, 0), 2);
      cv::rectangle(src_img, object_2d_rect, cv::Scalar(0, 150, 255), 2);
      // 0-blue_yellow, 1-blue_white, 2-blue_blue, 3-red_yellow, 4-red_white,
      // 5-red_red
      cv::putText(src_img,
                  std::to_string((int)res[yolo_res_selected_id].class_id),
                  cv::Point(object_2d_rect.x, object_2d_rect.y - 1),
                  cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 150, 255), 1);

      std::vector<uchar> buff_side_camera;
      cv::imencode(".jpg", src_img, buff_side_camera);
      streamer_ptr->publish("/sidecamera",
                            std::string(buff_side_camera.begin(),
                            buff_side_camera.end()));
#endif
      }
    } catch (const std::exception &e) {
    fmt::print("{}\n", e.what());
  }
}

void uartReadThread(const std::shared_ptr<RoboSerial> &serial,
                    RoboInf &robo_inf) {
  while (true) try {
      serial->ReceiveInfo(robo_inf);
      std::this_thread::sleep_for(1ms);
    } catch (const std::exception &e) {
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        std::this_thread::sleep_for(10000ms);
        fmt::print("[{}] read serial excepted to many times, sleep 10s.\n",
                   idntifier_red);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] serial exception: {} serial restarting...\n",
                 idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void uartThread(RoboCmd &robo_cmd, RoboInf &robo_inf, const std::shared_ptr<RoboSerial> &serial) {
  std::thread uart_read_thread(uartReadThread, std::ref(serial), std::ref(robo_inf));
  uart_read_thread.detach();

  while (true) try {
      serial->available();
    } catch (const std::exception &e) {
      static int change_serial_port_times{0};
      if (change_serial_port_times++ > 3) {
        fmt::print("[{}] Serial restarted to many times, sleep 10s...\n",
                   idntifier_red);
        std::this_thread::sleep_for(10000ms);
        change_serial_port_times = 0;
      }
      fmt::print("[{}] exception: {} serial restarting...\n", idntifier_red,
                 e.what());
      std::string port = serial->getPort();
      port.pop_back();
      try {
        serial->close();
      } catch (...) {
      }

      for (int i = 0; i < 3; i++) {
        fmt::print("[{}] try to change to {}{} port.\n", idntifier_red, port,
                   i);
        try {
          serial->setPort(port + std::to_string(i));
          serial->open();
        } catch (const std::exception &e1) {
          fmt::print("[{}] change {}{} serial failed.\n", idntifier_red, port,
                     i);
        }
        if (serial->isOpen()) {
          fmt::print("[{}] change to {}{} serial successed.\n", idntifier_green,
                     port, i);
          break;
        }
        std::this_thread::sleep_for(300ms);
      }
      if (serial->isOpen()) break;
    }
    std::this_thread::sleep_for(1000ms);
}

int main(int argc, char *argv[]) {
  RoboCmd robo_cmd;
  RoboInf robo_inf;

  auto streamer_ptr = std::make_shared<nadjieb::MJPEGStreamer>();
  streamer_ptr->start(8080);

  auto serial = std::make_shared<RoboSerial>("/dev/ttyUSB0", 115200);

  std::thread uart_thread(uartThread, std::ref(robo_cmd), std::ref(robo_inf), std::ref(serial));
  uart_thread.detach();

  std::thread top_camera_thread(topCameraThread, std::ref(robo_cmd),
                                std::ref(robo_inf), std::ref(serial));
  top_camera_thread.detach();

  std::thread side_camera_thread(sideCameraThread, std::ref(robo_cmd),
                                 std::ref(robo_inf), std::ref(serial),
                                 std::ref(streamer_ptr));
  side_camera_thread.detach();

  if (std::cin.get() == 'q') {
    top_camera_thread.~thread();
    side_camera_thread.~thread();
  }

  return 0;
}