#include "tracker.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <algorithm>
#define RELEASE

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
#ifndef RELEASE
void printPt(const Point& p) {
	char d=',';
	std::cout<<p.x<<d<<p.y<<std::endl;
}
void printPtf(const Point2f& p) {
	char d=',';
	std::cout<<p.x<<d<<p.y<<std::endl;
}
void printRt(const Rect& r) {
	char d=',';
	std::cout<<r.x<<d<<r.y<<d<<r.width<<d<<r.height<<std::endl;
}
#endif
void fback(Mat& pre, Mat& cur, Mat& flow) {
	Mat old,gray,uflow;
	cv::cvtColor(pre,old,COLOR_BGR2GRAY);
	cv::cvtColor(cur,gray,COLOR_BGR2GRAY);
	cv::calcOpticalFlowFarneback(old,gray,uflow,0.5,3,15,3,5,1.2,0);
	uflow.copyTo(flow);
}

void SurfTrack::UpdateDescriptor() {
	detector_->setHessianThreshold(minHessian_);
	detector_->detectAndCompute(simg_,UMat(),skps_,sdsp_);
	sCorners_[0] = cvPoint(0,0);
	sCorners_[1] = cvPoint(simg_.cols,0);
	sCorners_[2] = cvPoint(simg_.cols,simg_.rows);
	sCorners_[3] = cvPoint(0,simg_.rows);
}

bool SurfTrack::Match(Mat& img) {
#ifndef RELEASE
	cv::imshow("main",img);
	cv::waitKey(0);
#endif
	std::vector<cv::KeyPoint> kps;
	Mat dsp;
	detector_->detectAndCompute(img,UMat(),kps,dsp);
	vector<DMatch> matches;
	matcher_.match(sdsp_,dsp,matches);
	if (matches.size() == 0) return false;
	auto cmp = [](const DMatch& x,const DMatch& y){return x.distance < y.distance;};
	auto min_dist = std::min_element(begin(matches),end(matches),cmp)->distance;
	vector<DMatch> good_matches;
	for (const auto& m : matches) {
		if (m.distance < 3*min_dist) good_matches.push_back(m);
	}
	if (good_matches.size() == 0) return false;
	std::vector<Point2f> obj;
  std::vector<Point2f> scene;
	for (const auto& m : good_matches) {
		obj.push_back(skps_[m.queryIdx].pt);
		scene.push_back(kps[m.trainIdx].pt);
	}
  Mat H = findHomography(obj,scene,cv::RANSAC);
	if (H.empty()) return false;
#ifndef RELEASE
	std::cout << "get Homography" << H.rows <<','<<H.cols <<std::endl;
	//std::cout <<"sCorners_"<<std::endl;
	//for (const auto& p : sCorners_) printPtf(p);
#endif
	perspectiveTransform(sCorners_,tCorners_,H);
#ifndef RELEASE
	//std::cout << "get corners" << std::endl;
#endif
	minEnclosingCircle(tCorners_,center_,radius_);
#ifndef RELEASE
	cv::circle(img,center_,radius_,cv::Scalar(0,255,0),2);
	std::cout << "center:";printPtf(center_);
	std::cout << "radius: "<<radius_ << std::endl;
	imshow("main",img);
	cv::waitKey(0);
#endif
	TransformV2P();
	if (center_.inside(vzone_)) return true;
	else {
		std::cout << "center out" << std::endl;
		return false;
	}
}

void SurfTrack::InitTracker(const Mat& sample, const Rect& roi) {
	pzone_ = cv::Rect{cv::Point{0,0},sample.size()};
	sample(roi).copyTo(simg_); center_ = GetRectCenter(roi); scrop_ = roi;
	cvtColor(simg_,simg_,COLOR_BGR2GRAY);
	sideLen_ = max(roi.width,roi.height);
	UpdateDescriptor();
	UpdateVZone();
}

bool SurfTrack::Track(const Mat& scene) {
	Mat roi;scene(vzone_).copyTo(roi);
	cvtColor(roi,roi,COLOR_BGR2GRAY);
	auto ret = Match(roi);
	if (ret) UpdateVZone();
	return ret;
}

SurfTrack::SurfTrack(float sideScale, double minHessian):
	sideScale_{sideScale},sCorners_(4),tCorners_(4),
	matcher_{},minHessian_{minHessian} {
	detector_ = cv::xfeatures2d::SURF::create(minHessian);
}
