#pragma once
#include <atomic>

enum CatchMode {
  wait = 0,
  spin,
  go,
  catch_cube
};

struct RoboInf {
  std::atomic<bool> auto_catch_cube_mode {false}; // 自动模式（自动对位并进行识别）
  std::atomic<bool> manual_catch_cube_mode {false}; // 手动模式（没有自动对位）
  std::atomic<bool> detect_cube_mode {false}; // 识别立起积木模式（仅进行积木状态识别）
  std::atomic<CatchMode> catch_cube_mode_status {CatchMode::wait};
};

// send R2 spin command
struct RoboSpinCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x01;
  float yaw_angle = 0.f;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 spin command
struct RoboGoCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x02;
  float distance = 0.f;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 catch command
// cube_state: 0x01 - yellow, 0x02 - white, 0x03 - stand
// cube_type: 0x01 - 0x05
struct RoboCatchCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x03;
  uint8_t cube_state = 0x00;
  uint8_t cube_type = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 cube status
// 0x01 white 0x02 yellow
// cube_type: 0x01 - 0x05
struct RoboCubeStateUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = 0x04;
  uint8_t cube_status = 0x00;
  uint8_t cube_type = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//uart recive
struct RoboInfUartBuff {
  bool auto_catch_cube_mode {false};
  bool manual_catch_cube_mode {false};
  bool detect_cube_mode {false};
} __attribute__((packed));

//选择在图像 x 轴 1/3 - 2/3 范围中，距离夹子最近的积木
bool rectFilter(std::vector<Yolo::Detection> res, cv::Mat &img,
                cv::Rect &rect, int &select_id) {
  std::vector<Yolo::Detection> middle_res;
  for (size_t i = 0; i < res.size(); i++) {
    if (res[i].bbox[1] > img.cols * 0.3 && res[i].bbox[1] < img.cols * 0.6) {
      middle_res.emplace_back(res[i]);
    }
  }
  
  float max_y_axis = .0;
  select_id = -1;
  for (size_t i = 0; i < middle_res.size(); i++) {
    if (middle_res[i].bbox[1] > max_y_axis) {
      max_y_axis = middle_res[i].bbox[1];
      select_id = i;
    }
  }
  if (select_id != -1) {
    rect = get_rect(img, middle_res[select_id].bbox);
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
