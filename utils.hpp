#pragma once
#include <atomic>
#include "OpenVINO/detector.h"

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

#define CUBE_LIE    0x01
#define CUBE_ERECT  0x02
#define CUBE_STAND 0x03

#define SPIN_SIGN         0x01
#define GO_SIGN           0X02
#define CATCH_SIGN        0X03
#define RETURN_CUBE_STATE 0X04

#define WATTING 0x01

enum CatchMode {
  off = 0,
  spin,
  go,
  catch_cube,
  detect_mode
};

struct RoboInf {
  std::atomic<uint8_t> mode {0x00};
  std::atomic<CatchMode> catch_cube_mode_status {CatchMode::off};
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
  uint8_t E_flag = 'E';
} __attribute__((packed));

// send R2 cube status
// 0x01 white 0x02 yellow
// cube_type: 0x01 - 0x05
struct RoboCubeStateUartBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = RETURN_CUBE_STATE;
  uint8_t cube_status = 0x00;
  uint8_t E_flag = 'E';
} __attribute__((packed));

//uart recive
struct RoboInfUartBuff {
  uint8_t mode = NOTHING;
} __attribute__((packed));

/**
 * @brief 筛选出距离夹子最近的 rect
 * 
 * @param obs openvino 识别结果
 * @param src_img 识别结果的原图像
 * @param selected_rect 选中的目标
 * @return true 
 * @return false 
 */
bool rectFilter(const std::vector<Detector::Object> &objs,
                const cv::Mat &src_img,
                Detector::Object &selected_rect) {
  std::vector<Detector::Object> catch_range_objs; // 在夹子范围内的 rect
  for (auto obj : objs) {
    if ((obj.rect.y + obj.rect.width / 2) > (src_img.cols / 3) &&
        (obj.rect.x + obj.rect.width / 2) < (src_img.cols / 3 * 2)) {
          catch_range_objs.emplace_back(obj);
        }
  }

  std::sort(catch_range_objs.begin(), catch_range_objs.end(),
            [](Detector::Object oj_i, Detector::Object oj_j)
            { return (oj_i.rect.y > oj_j.rect.y); });

  if (catch_range_objs.size() != 0) {
    selected_rect = catch_range_objs[0];
    return true;
  } else {
    return false;
  }
}



// serial change
struct RobotRcNBuff {
  uint8_t S_flag = 'S';
  uint8_t cmd_type = WATTING;  // Grippers
  uint8_t cube_status = 0x00;  // the type of the cube
  float distance = 0.f;  // distance
  float yaw = 0.f;  // yaw
  uint8_t E_flag = 'E';
} __attribute__((packed));
