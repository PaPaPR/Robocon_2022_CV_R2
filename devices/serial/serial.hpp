#pragma once
#include "serial/serial.h"
#include "utils.hpp"

auto idntifier_green = fmt::format(fg(fmt::color::green) | fmt::emphasis::bold, "serial");
auto idntifier_red   = fmt::format(fg(fmt::color::red)   | fmt::emphasis::bold, "serial");

class RoboSerial : public serial::Serial {
 public:
  RoboSerial(std::string port, unsigned long baud) {
    auto timeout = serial::Timeout::simpleTimeout(serial::Timeout::max());
    this->setPort(port);
    this->setBaudrate(baud);
    this->setTimeout(timeout);
    try {
      this->open();
      fmt::print("[{}] Serial init successed.\n", idntifier_green);
    } catch(const std::exception& e) {
      fmt::print("[{}] Serial init failed, {}.\n", idntifier_red, e.what());
    }
  }

  void WriteInfo(RoboCmd &robo_cmd) {
    RoboCmdUartBuff t1;
    t1.yaw_angle = robo_cmd.yaw_angle.load();
    t1.pitch_angle = robo_cmd.pitch_angle.load();
    t1.depth = robo_cmd.depth.load();
    t1.detect_object = robo_cmd.detect_object.load();

    this->write((uint8_t *)&t1, sizeof(t1));
  }

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInfUartBuff uart_buff_struct;
    uint8_t uart_S_flag;
    this->read(&uart_S_flag, 1);
    while (uart_S_flag != 'S')
      this->read(&uart_S_flag, 1);
    this->read((uint8_t *)&uart_buff_struct, sizeof(uart_buff_struct));
    robo_inf.catch_cube_flag.store(uart_buff_struct.catch_cube_mode);
    robo_inf.detect_cube_mode.store(uart_buff_struct.detect_cube_mode);
  }

 private:
};