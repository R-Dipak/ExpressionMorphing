#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include<dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <cv.h>
#include <highgui.h>
typedef dlib::full_object_detection fod;
#define eps 1
#define pii pair<int,int>
//using namespace dlib;
using namespace std;
using namespace cv;

struct Triangle{
	int id1,id2, id3;
	Triangle(int a,int b, int c){
		id1= a;id2= b; id3= c;}
};
vector<Triangle> Tlist;

class Face{
	public:
		int length;
	Mat img;
	vector<Vec6f> Trilist;
	dlib::full_object_detection shape;        // Landmark points
	vector<Point2f> shape2;
	void New(){
		Trilist.clear();
		shape2.clear();
	}
	Subdiv2D subdiv;
	void Create_morphed_landMark();//const fod &Normal, const fod &express, const fod &tobpr);
	void DelaunayTriangulate(int fl);

	void equalize(int i =0){
		if(i==0){
			for(int i =0; i< 68; i++){
				Point2f pont(shape.part(i).x(), shape.part(i).y());
				shape2.push_back(pont);
			}
		}
	}
	int findindex(Point2f pt){
		int ans = -1;
		for(int i =0; i< shape2.size(); i++){
			if(abs(shape2[i].x-pt.x)<eps && abs(shape2[i].y-pt.y)<eps)
				return i;
		}
		return ans;
	}

   void calcTriangles(vector<Point2f> &srcTri, Face & src, int i){
      Vec6f s = src.Trilist[i];
		srcTri.push_back( Point(cvRound(s[0]), cvRound(s[1])));
		srcTri.push_back ( Point(cvRound(s[2]), cvRound(s[3])));
		srcTri.push_back(Point(cvRound(s[4]), cvRound(s[5])));
	}
	void AffineTransform();

}Faces[4];

void Face::AffineTransform(){
//	imshow("B4 affine",Faces[3].img);waitKey(0);
	Mat rot_mat( 2, 3, CV_32FC1 );
	Mat warp_mat( 2, 3, CV_32FC1 );
	Mat imgo = Faces[3].img.clone();
	Mat Totkernel = Mat(img.rows,img.cols,CV_8UC1,Scalar(255));
	set<pii > Used;
	Used.clear();
	for(int i =0; i< Tlist.size(); i++){
	   vector<Point2f> srcTri,dstTri;int a = Tlist[i].id1, b = Tlist[i].id2, c=  Tlist[i].id3;
		pii l1(a,b),l2(b,c),l3(c,a);
	line(imgo, Faces[3].shape2[a], Faces[3].shape2[b], Scalar(0,0,255), 1);
	line(imgo, Faces[3].shape2[a], Faces[3].shape2[c], Scalar(0,0,255), 1);
	line(imgo, Faces[3].shape2[c], Faces[3].shape2[b], Scalar(0,0,255), 1);
	if(Used.find(l1) != Used.end() || Used.find(pii(b,a))!= Used.end())
				line(img, Faces[3].shape2[a], Faces[3].shape2[b], Scalar(0,0,0), 1);// CV_AA, 0);

		if(Used.find(l2) != Used.end() || Used.find(pii(c,b))!= Used.end())
				line(img, Faces[3].shape2[b], Faces[3].shape2[c], Scalar(0,0,0), 1);//, CV_AA, 0);

		if(Used.find(l3) != Used.end() || Used.find(pii(a,c))!= Used.end())
				line(img, Faces[3].shape2[c], Faces[3].shape2[a], Scalar(0,0,0), 1);//, CV_AA, 0);
		srcTri.push_back(Faces[2].shape2[a]);
		srcTri.push_back(Faces[2].shape2[b]);
		srcTri.push_back(Faces[2].shape2[c]);
		dstTri.push_back(Faces[3].shape2[a]);
		dstTri.push_back(Faces[3].shape2[b]);
		dstTri.push_back(Faces[3].shape2[c]);
		Used.insert(l1);Used.insert(l2); Used.insert(l3);
		{
		Rect srcrct = boundingRect( (srcTri)	),dstrct = boundingRect ( (dstTri)) ;	
     	Point dstr1[3] = {dstTri[0],dstTri[1], dstTri[2]}; 
		fillConvexPoly(Totkernel,dstr1,3,Scalar(0));
		
		srcTri[0].x-=srcrct.x;srcTri[0].y-=srcrct.y;
		srcTri[1].x-=srcrct.x;srcTri[1].y-=srcrct.y;
		srcTri[2].x-=srcrct.x;srcTri[2].y-=srcrct.y;
		dstTri[0].x-=dstrct.x;dstTri[0].y-=dstrct.y;
		dstTri[1].x-=dstrct.x;dstTri[1].y-=dstrct.y;
		dstTri[2].x-=dstrct.x;dstTri[2].y-=dstrct.y;
		Mat imga = Faces[2].img(srcrct).clone(),imgb = Faces[3].img(dstrct).clone();
		warp_mat = getAffineTransform( srcTri,dstTri );

		warpAffine( imga, imgb, warp_mat, imgb.size(),BORDER_REFLECT_101);
		Mat kernel = Mat::zeros(imgb.rows,imgb.cols, CV_8UC1);
		Point dstr[3] = {dstTri[0], dstTri[1], dstTri[2]};
		fillConvexPoly(kernel,dstr,3,Scalar(255));
		add( Faces[3].img(dstrct), imgb,Faces[3].img(dstrct), kernel);

		}
	}
	add(Faces[2].img, Faces[3].img, Faces[3].img, Totkernel);
	medianBlur(Faces[3].img, Faces[3].img,3);
	imwrite("Thedelop.png",imgo);
	imwrite("MorphedImage.png",img); 
	imshow("Original",Faces[2].img);
	waitKey(0);
	imshow( "Display window",img );
   waitKey(0);
}

void Face::DelaunayTriangulate(int fl){
	Rect rect(0, 0, img.size().width, img.size().height);
	Subdiv2D subdiv(rect); 
	for(int i =0; i< shape2.size(); i++){
		if(rect.contains(shape2[i]))
			subdiv.insert(shape2[i]);
	
	}
	this->subdiv= subdiv;
	subdiv.getTriangleList(Trilist);
	
	Mat img1 = img.clone();

	for(int i =0; i< Trilist.size(); i++){
		Vec6f t = Trilist[i];int a,b,c;
		Point pt[3];
		pt[0] = Point2f(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point2f(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point2f(cvRound(t[4]), cvRound(t[5]));
		if(rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{  

		a = findindex(pt[0]);
		b = findindex(pt[1]);
		c = findindex(pt[2]);Triangle now(a,b,c);
		Tlist.push_back(now);
			line(img1, pt[0], pt[1], Scalar(255,0,255), 1, CV_AA, 0);
			line(img1, pt[1], pt[2], Scalar(255,0,255), 1, CV_AA, 0);
			line(img1, pt[2], pt[0], Scalar(255,0,255), 1, CV_AA, 0);
		}
	}
	imwrite("del.png",img1);
}

void Face:: Create_morphed_landMark(){ //const fod &Normal, const fod &express, const fod &tobpr){
	
	double factor = Faces[2].length/(double)Faces[0].length;
	for(int i =0; i< Faces[2].shape2.size(); i++){
		Point2f pont;
		if(i<28){
				 pont = Point2f(Faces[2].shape2[i].x,Faces[2].shape2[i].y );
		}
		else
		 pont = Point2f(Faces[2].shape2[i].x+factor*(Faces[1].shape2[i].x-Faces[0].shape2[i].x),
		Faces[2].shape2[i].y+factor*(Faces[1].shape2[i].y-Faces[0].shape2[i].y) );
		Faces[3].shape2.push_back(pont);
	}
	imwrite("Points.png", Faces[3].img);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

	if (argc == 1)
	{
		cout << "Call this program like this:" << endl;
		cout << "./Srcforimages shape_predictor_68_face_landmarks.dat ";
		cout << "NeutralFace1 ExpressionFace1 NeutralFace2 "<<endl;
		return 0;
	}

	Faces[0].img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);                                 // Take in all the images, neutral1
	Faces[1].img = cv::imread(argv[3],CV_LOAD_IMAGE_COLOR);											// Smile1
   Faces[2].img = cv::imread(argv[4],CV_LOAD_IMAGE_COLOR);                                 // Neutral2
   Faces[3].img = Mat::zeros(Faces[2].img.rows, Faces[2].img.cols, CV_8UC3);               // Tobsmile2

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize(argv[1]) >> sp;


	for (int i = 2; i <= 4; ++i)
	{
		dlib::cv_image<dlib::bgr_pixel> img(Faces[i-2].img);
		std::vector<dlib::rectangle> dets = detector(img);                  // dets bounds the faces with rectangles

		std::vector<dlib::full_object_detection> shapes;
		dlib::full_object_detection shape = sp(img, dets[0]);
		Faces[i-2].shape = shape;
		shapes.push_back(shape);
		Faces[i-2].equalize();
		Faces[i-2].length = dets[0].height();
	}


		Faces[3].Create_morphed_landMark();

		Faces[2].DelaunayTriangulate(0);
		Faces[3].AffineTransform();
}
