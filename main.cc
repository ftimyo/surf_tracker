#include <iostream>
#include <string>
#include "tracker.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

struct Param {
	Param():pause{false},draw{false}{}
	bool pause;
	bool draw;
	cv::Rect rect;
	cv::Mat img;
};
void switchpause(int, void* data) {
	Param* p = (Param*)data;
	p->pause = !p->pause;p->draw = false;
}
void onMouse(int event, int x, int y, int, void* userInput) {
	auto pm = static_cast<Param*>(userInput);
	//if (!pm->pause) return;
	auto m = pm->img.clone();
	if (event == cv::EVENT_LBUTTONDOWN) {
		pm->rect = cv::Rect{x,y,0,0};
		pm->draw = true;
	}
	if (event == cv::EVENT_MOUSEMOVE && pm->draw) {
		auto tl = pm->rect.tl();
		pm->rect = cv::Rect{tl,cv::Point{x,y}};
		cv::rectangle(m,pm->rect,cv::Scalar(0,255,0),2);
	}
	if (event == cv::EVENT_LBUTTONUP) {
		auto tl = pm->rect.tl();
		pm->rect = cv::Rect{tl,cv::Point{x,y}};
		pm->draw = false;
		cv::rectangle(m,pm->rect,cv::Scalar(0,255,0),2);
	}
	cv::imshow("main",m);
}

int main(int, char* argv[]) {
	//Param pm;
	cv::namedWindow("main",CV_GUI_EXPANDED|CV_WINDOW_KEEPRATIO);
	cv::namedWindow("main2",CV_GUI_EXPANDED|CV_WINDOW_KEEPRATIO);
//	cv::setMouseCallback("main",onMouse,&pm);
	cv::VideoCapture vs{argv[1]};
	SurfTrack tracker{3,400};
	cv::Mat img;
	vs.read(img);
	//tracker.InitTracker(img,cv::Rect{2516,1095,122,54}); //for sample.mp4 4K
	//tracker.InitTracker(img,cv::Rect{1259,549,60,27}); //for sample2.mp4 2K
	//tracker.InitTracker(img,cv::Rect{3122,1098,180,80}); //for 65 meter sample6.mp4 4K
	tracker.InitTracker(img,cv::Rect{1561,549,90,40}); //for 65 meter sample602.mp4 2K
	//tracker.InitTracker(img,cv::Rect{780,275,45,20}); //for 65 meter sample603.mp4 1K
	for (const auto& pt : tracker.skps_) {
		cv::Point pxt{cvRound(pt.pt.x),cvRound(pt.pt.y)};
		cv::circle(tracker.simg_,pxt,0,cv::Scalar(255,0,0),1);
	}
	cv::imshow("main",tracker.simg_);
	cv::waitKey(0);
	while (vs.read(img)) {
		auto timer = std::chrono::high_resolution_clock::now();
		auto ret = tracker.Track(img);
		auto t = std::chrono::duration_cast<std::chrono::duration<double>>(
				std::chrono::high_resolution_clock::now() - timer);
		std::cout << t.count() << std::endl;
		if (ret) {
			cv::Point center; float r=0;
			tracker.GetCircle(center,r);
			//cv::circle(img,center,r,cv::Scalar(0,255,0),10);
			for (const auto& pt : tracker.mfpt_) {
				cv::Point pppt {cvRound(pt.x),cvRound(pt.y)};
				cv::circle(img,pppt,0,cv::Scalar(255,0,0),1);
			}
			cv::imshow("main",img);
			cv::imshow("main2",tracker.simg_);
			cv::waitKey(0);
		} else {std::cerr << "track fail" << std::endl;}
	}
	return 0;
}
