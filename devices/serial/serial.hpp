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

  void ReceiveInfo(RoboInf &robo_inf) {
    RoboInfUartBuff uart_buff_struct;
    uint8_t uart_S_flag;
    this->read(&uart_S_flag, 1);
    while (uart_S_flag != 'S')
      this->read(&uart_S_flag, 1);
    this->read((uint8_t *)&uart_buff_struct, sizeof(uart_buff_struct));
    robo_inf.auto_catch_cube_mode.store(uart_buff_struct.auto_catch_cube_mode);
    robo_inf.manual_catch_cube_mode.store(uart_buff_struct.manual_catch_cube_mode);
    robo_inf.detect_cube_mode.store(uart_buff_struct.detect_cube_mode);
  }

 private:
};