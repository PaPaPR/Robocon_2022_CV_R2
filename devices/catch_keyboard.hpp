// need libevdev-dev package and root permission
#pragma once
#include <fcntl.h>
#include <libevdev-1.0/libevdev/libevdev.h>
#include <linux/input.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <exception>
#include <iostream>

class CatchKeyboard {
 private:
  char kb_name_[BUFSIZ];
  char kb_path_[BUFSIZ];
  bool is_open_{false};
  int fd_;
  struct libevdev *dev_;

 public:
  CatchKeyboard(const char* keyboard_name);
  bool FindKeyboard();
  bool OpenKeyboard();
  bool GetEvent(unsigned short &code);
  ~CatchKeyboard();
};

CatchKeyboard::CatchKeyboard(const char* keyboard_name) {
  strcpy(kb_name_, keyboard_name);
  this->FindKeyboard();
  if (this->OpenKeyboard()){
    std::cout << "keyboard open success.\n";
  } else {
    std::cout << "keyboard open failed.\n";
  }
}

CatchKeyboard::~CatchKeyboard() {
  close(fd_);
}

bool CatchKeyboard::FindKeyboard() {
  char kb_name_buf[BUFSIZ];
  char kb_path_buf[BUFSIZ];
  bool flag{false};
  for (int i = 0; i < 50; i++) {
    sprintf(kb_path_buf, "/dev/input/event%d", i);
    try {
      if ((fd_ = open(kb_path_buf, O_RDONLY)) < 0) break;
      ioctl(fd_, EVIOCGNAME(sizeof(kb_name_buf)), kb_name_buf);
    } catch (const std::exception& e) {
      std::cout << e.what() << '\n';
    }

    if (strstr(kb_name_buf, kb_name_)) {
      strcpy(kb_path_, kb_path_buf);
      flag = true;
      break;
    }
    close(fd_);
  }
  return flag;
}

bool CatchKeyboard::OpenKeyboard() {
  fd_ = open(kb_path_, O_RDONLY | O_NONBLOCK);
  int err = libevdev_new_from_fd(fd_, &dev_);
  if (fd_ >= 0 && err >= 0) {
    is_open_ = true;
  }
  return is_open_;
}

bool CatchKeyboard::GetEvent(unsigned short &code) {
  bool rt{false};
  if (is_open_) {
    struct input_event ev;
    int err = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_NORMAL, &ev);
    if (err == 0 && ev.type == EV_KEY) {
      // printf("Event: %s %s %d\n", libevdev_event_type_get_name(ev.type),
      //         libevdev_event_code_get_name(ev.type, ev.code), ev.value);
      code = ev.code;
      if (ev.value == 1) {
        rt = true;
      }
    }
  }
  return rt;
}