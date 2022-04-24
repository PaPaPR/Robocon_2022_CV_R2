#include "videoRecord.h"

struct Record_counts
{
 int id = 0;
 int count_ = 0;
 int interval = 5; // 时间间隔
};

void recordVideo(Record_counts &counts_){
 rs2::pipeline pipe;
 rs2::config cfg;
 cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30.f);
 cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30.f);
 pipe.start(cfg);
 rs2::align align_to(RS2_STREAM_COLOR);
 cv::Mat src_img;
 while (true)try
 {
  auto frames = pipe.wait_for_frames();
  auto depth_frame = frames.get_depth_frame();
  auto aligned_set = align_to.process(frames);
  auto color_frame = aligned_set.get_color_frame();
  src_img = frame_to_mat(color_frame);
  cv::cvtColor(src_img,src_img,cv::COLOR_BGR2RGB);
  if(cv::waitKey(1) == 'q') {
   break;
  }
  if(cv::waitKey(1) == 's' || counts_.count_ % 10 == 0){
   cv::imwrite("../Image_Save" + std::to_string(++counts_.id) + ".jpg",src_img);
   fmt::print(fmt::format(fg(fmt::color::green)|fmt::emphasis::bold, "SAVE[{}]",counts_.id));
  }
  cv::putText(src_img,
             std::to_string(counts_.id),
             cv::Point(100,100),
             3,
             cv::FONT_HERSHEY_DUPLEX,
             cv::Scalar(0,0,0),
             3,
             cv::LINE_8
             );
 }
 catch(const std::exception& e)
 {
  std::cerr << e.what() << '\n';
 }
}

void countBegin(Record_counts &counts_)
{
 while (true)try
 {
  std::this_thread::sleep_for(std::chrono::seconds(counts_.interval));
  counts_.count_ += counts_.interval;
  
 }
 catch(const std::exception& e)
 {
  std::cerr << e.what() << '\n';
 }
 
}
int main()
{
 Record_counts counts_obj;
 std::thread Videothreard(recordVideo,std::ref(counts_obj));
 std::thread Countsthread(countBegin,std::ref(counts_obj));
 if(Videothreard.joinable()){
  Videothreard.detach();
 }
 if(Countsthread.joinable()){
  Countsthread.detach();
 }
 if(cv::waitKey(0) == 'q') {
  return 0;
 }

}