#pragma once
#include <atomic>

enum CatchMode {
  wait = 0,
  spin,
  go,
  catch_cube
};

struct RoboCmd {
  std::atomic<float> pitch_angle = 0.f;
  std::atomic<float> yaw_angle = 0.f;
  std::atomic<float> depth = 0.f;
  std::atomic<uint8_t> detect_object {false};
};

struct RoboInf {
  std::atomic<bool> catch_cube_mode {false};
  std::atomic<bool> detect_cube_mode {false};
  std::atomic<CatchMode> catch_cube_mode_status {CatchMode::wait};
};

//structs below are used to uart to send and recive info
struct RoboCmdUartBuff {
  uint8_t S_flag = 'S';
  float pitch_angle = 0.f;
  float yaw_angle = 0.f;
  float depth = 0.f;
  uint8_t detect_object {false};
  uint8_t E_flag = 'E';
} __attribute__((packed));

//send R2 spin command
struct RoboSpinCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x01;
  float yaw_angle = 0;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//send R2 spin command
struct RoboGoCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x02;
  float distance = 0.f;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//send R2 catch command
// cube_state: 0x01 - yellow, 0x02 - white, 0x03 - stand
struct RoboCatchCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x03;
  uint8_t cube_state = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//send R2 cube status
//0x01 white 0x02 yellow
struct RoboCubeStateUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cube_status = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//uart recive
struct RoboInfUartBuff {
  bool catch_cube_mode {false};
  bool detect_cube_mode {false};
} __attribute__((packed));

bool rectFilter(std::vector<Yolo::Detection> res, cv::Mat &img,
                cv::Rect &rect, int &select_id) {
  float max_x_axis = .0;
  select_id = -1;
  for (size_t i = 0; i < res.size(); i++) {
    if (res[i].bbox[1] > max_x_axis) {
      max_x_axis = res[i].bbox[1];
      select_id = i;
    }
  }
  if (select_id != -1) {
    rect = get_rect(img, res[select_id].bbox);
    return true;
  } else {
    return false;
  }
}

bool sideRectFilter(std::vector<Yolo::Detection> res, cv::Mat &img,
                    cv::Rect &rect, int &select_id) {
  select_id = -1;
  for (size_t i = 0; i < res.size(); i++) {
    if (res[i].bbox[0] > img.rows / 3 && res[i].bbox[0] > img.rows / 3 * 2 &&
        res[i].bbox[1] > img.cols / 3 && res[i].bbox[1] > img.cols / 3 * 2)
    {
      select_id = i;
    }
  }
  if (select_id != -1) {
    rect = get_rect(img, res[select_id].bbox);
    return true;
  } else {
    return false;
  }
}
