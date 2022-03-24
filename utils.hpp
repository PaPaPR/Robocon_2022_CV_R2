#pragma once
#include <atomic>

#define AUTO_MODE 0x01
#define MANUAL_MODE 0x02
#define DETECT_MODE 0x03
#define NOTHING 0x04

#define CUBE_1 0x01 // cube_1(biggest)
#define CUBE_2 0x02
#define CUBE_3 0x03
#define CUBE_4 0x04
#define CUBE_5 0x05
#define CUBE_UNCERTAIN 0X06
#define CUBE_UP    0x01
#define CUBE_DOWN  0x02
#define CUBE_STAND 0x03

#define SPIN_SIGN         0x01
#define GO_SIGN           0X02
#define CATCH_SIGN        0X03
#define RETURN_CUBE_STATE 0X04

enum CatchMode {
  wait = 0,
  spin,
  go,
  catch_cube
};

struct RoboInf {
  std::atomic<uint8_t> mode {0x00};
  std::atomic<CatchMode> catch_cube_mode_status {CatchMode::wait};
};

// send R2 spin command
struct RoboSpinCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = SPIN_SIGN;
  float yaw_angle = 0.f;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 spin command
struct RoboGoCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = GO_SIGN;
  float distance = 0.f;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 catch command
// cube_state: 0x01 - yellow, 0x02 - white, 0x03 - stand
// cube_type: 0x01 - 0x05
struct RoboCatchCmdUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = CATCH_SIGN;
  uint8_t cube_state = 0x00;
  uint8_t cube_type = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 cube status
// 0x01 white 0x02 yellow
// cube_type: 0x01 - 0x05
struct RoboCubeStateUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = RETURN_CUBE_STATE;
  uint8_t cube_status = 0x00;
  uint8_t cube_type = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//uart recive
struct RoboInfUartBuff {
  uint8_t mode = NOTHING;
} __attribute__((packed));

//选择在图像 x 轴 1/3 - 2/3 范围中，距离夹子最近的积木
bool rectFilter(std::vector<Yolo::Detection> res, cv::Mat &img,
                cv::Rect &rect, int &select_id) {
  std::vector<int> middle_res_no;
  cv::Rect rc;
  for (size_t i = 0; i < res.size(); i++) {
    rc = get_rect(img, res[i].bbox);
    if ((rc.x + rc.width / 2) > (img.cols / 3) && (rc.x + rc.width / 2) < (img.cols / 3 * 2)) {
      middle_res_no.emplace_back(i);
    }
  }
  
  float max_y_axis = .0;
  select_id = -1;
  for (size_t i = 0; i < middle_res_no.size(); i++) {
    if (res[middle_res_no[i]].bbox[1] > max_y_axis) {
      max_y_axis = res[middle_res_no[i]].bbox[1];
      select_id = middle_res_no[i];
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
