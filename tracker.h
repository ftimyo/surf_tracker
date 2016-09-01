#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <boost/shared_ptr.hpp>
class SurfTrack {
public:
	SurfTrack(float sideScale=3, double minHessian = 400);
	static cv::Point GetRectCenter(const cv::Rect& b) {return (b.tl() + b.br()) / 2;}
//private:
	cv::Mat simg_; //sample image
	cv::Mat sdsp_; //sample image descriptors
	std::vector<cv::KeyPoint> skps_; //sample image key points
	cv::Rect scrop_; //crop sample image
	int sideLen_; //sample image max(width,height)
	float sideScale_;//VZone size=(sideLen*sideScale_)^2
	cv::Rect pzone_;//whole image zone, physical coordinates
	cv::Rect vzone_;//virtual zone sideScale x sideScale sizeLen, physical coordinates
	std::vector<cv::Point2f> sCorners_; //sample corners
	std::vector<cv::Point2f> tCorners_; //tracked corners
	cv::FlannBasedMatcher matcher_;
	double minHessian_;
	cv::Ptr<cv::xfeatures2d::SURF> detector_;
	cv::Point2f center_;
	float radius_;

	void UpdateDescriptor();
	bool Match(cv::Mat&);

	inline void TransformV2P() {/*transform virtual coordinates to physical*/
		cv::Point2f vo = vzone_.tl();/*virtual origin*/ center_ += vo;
		for (auto& pt : tCorners_) pt += vo;
	}
	inline void UpdateVZone() {
		vzone_ = pzone_;
#if 1
		int len = sideScale_*sideLen_;
		cv::Rect zone{cv::Point{center_}-cv::Point{len>>1,len>>1},cv::Size{len,len}};
		vzone_ = pzone_ & zone;
#endif
	}
public:
	inline operator bool(){return !simg_.empty();}
	void InitTracker(const cv::Mat&,const cv::Rect&);
	bool Track(const cv::Mat&);
	inline void GetCircle(cv::Point& center,float radius){center=center_;radius=radius_;}
};
