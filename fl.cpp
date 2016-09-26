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
#define eps 10
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
	Mat img;
	vector<Vec6f> Trilist;
	dlib::full_object_detection shape;        // Landmark points
	vector<Point2f> shape2;
	Subdiv2D subdiv;
//	vector<Point2f> Create_morphed_landMark(const fod &Normal, const fod &express, const fod &tobpr);
	void DelaunayTriangulate(int fl);

	void equalize(int i =0){
		if(i==0){
			for(int i =0; i< 68; i++){
				Point2f pont(shape.part(i).x(), shape.part(i).y());
				shape2.push_back(pont);
			}
		}
		cout<<"equalised"<<endl;
	}
	int findindex(Point2f pt){
		int ans = -1;
		for(int i =0; i< shape2.size(); i++){
			if(abs(shape2[i].x-pt.x)<eps && abs(shape2[i].y-pt.y)<eps)
				return i;
		}
		return ans;
	}

//	void AffineTransform(Face &src,Face & dst);
   void calcTriangles(vector<Point2f> &srcTri, Face & src, int i){
      Vec6f s = src.Trilist[i];
		srcTri.push_back( Point(cvRound(s[0]), cvRound(s[1])));
		srcTri.push_back ( Point(cvRound(s[2]), cvRound(s[3])));
		srcTri.push_back(Point(cvRound(s[4]), cvRound(s[5])));
	}
//	void AffineTransform();
}Faces[4];

/*
void Face::AffineTransform(){
	Mat rot_mat( 2, 3, CV_32FC1 );
	Mat warp_mat( 2, 3, CV_32FC1 );
      vector<Point2f> srcTri,dstTri;
		srcTri.push_back(shape2[0]);
		srcTri.push_back(shape2[1]);
		srcTri.push_back(shape2[2]);
		dstTri.push_back(Point2f(shape2[0].x-10,shape2[0].y+6));
		dstTri.push_back(Point2f(shape2[1].x,shape2[1].y+2));
		srcTri.push_back(Point2f(shape2[2].x+10,shape2[0].y+6));

		{
		Rect srcrct = boundingRect( (srcTri)	),dstrct = boundingRect ( (dstTri)) ;	
      srcTri[0].x-=srcrct.x;srcTri[0].y-=srcrct.y;
		srcTri[1].x-=srcrct.x;srcTri[1].y-=srcrct.y;
		srcTri[2].x-=srcrct.x;srcTri[2].y-=srcrct.y;
		dstTri[0].x-=dstrct.x;dstTri[0].y-=dstrct.y;
		dstTri[1].x-=dstrct.x;dstTri[1].y-=dstrct.y;
		dstTri[2].x-=dstrct.x;dstTri[2].y-=dstrct.y;

		Mat imga = img(srcrct).clone(),imgb = img(dstrct).clone();
		warp_mat = getAffineTransform( srcTri, dstTri );

		warpAffine( imga, imgb, warp_mat, imgb.size() ,BORDER_REFLECT_101);
		Mat kernel = Mat::zeros(imgb.rows,imgb.cols, CV_8UC1);
		Point dstr[3] = {dstTri[0], dstTri[1], dstTri[2]};
		fillConvexPoly(kernel,dstr,3,Scalar(255));
		add(imgb, img(dstrct), img(dstrct), kernel);
		}
	
	namedWindow( "Display window", WINDOW_AUTOSIZE );
	imshow( "Display window",img );
   waitKey(0);
}
*/
void Face::DelaunayTriangulate(int fl){
	Rect rect(0, 0, img.size().width, img.size().height);
	Subdiv2D subdiv(rect); 
	for(int i =0; i< shape2.size(); i++){
		if(rect.contains(shape2[i]))
			subdiv.insert(shape2[i]);
	
	}
	this->subdiv= subdiv;
	subdiv.getTriangleList(Trilist);
	cout<<Trilist.size()<<endl;

	cout<<"No of triangles are "<<Trilist.size()<<endl;
	
	Mat img1 = img.clone();

	for(int i =0; i< Trilist.size(); i++){
		Vec6f t = Trilist[i];int a,b,c;
		Point pt[3];
		pt[0] = Point2f(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point2f(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point2f(cvRound(t[4]), cvRound(t[5]));
		if(rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{  
		//	cout<<i<<endl;

		a = findindex(pt[0]);
		b = findindex(pt[1]);
		c = findindex(pt[2]);Triangle now(a,b,c);
		Tlist.push_back(now);
//		cout<<a<<' '<<b<<' '<<c<<endl;
			line(img1, pt[0], pt[1], Scalar(255,0,255), 1, CV_AA, 0);
			line(img1, pt[1], pt[2], Scalar(255,0,255), 1, CV_AA, 0);
			line(img1, pt[2], pt[0], Scalar(255,0,255), 1, CV_AA, 0);
		}
		imshow("interimdel", img1);
		waitKey(250);
	}
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", img1);
	waitKey(0);
}

/*
vector<Point2f> Face:: Create_morphed_landMark(const fod &Normal, const fod &express, const fod &tobpr){
	vector<Point2f> ret;
	for(int i =0; i< tobpr.num_parts(); i++){
		Point2f pont(tobpr.part(i).x() + (express.part(i).x()-Normal.part(i).x()),
		tobpr.part(i).y() + (express.part(i).y()-Normal.part(i).y()));
		ret.push_back(pont);
	}
//	for(int i =0; i<68; i++){
	//	dlib::rgb_pixel a;
	//   dlib::draw_solid_circle ( img,shape.part(i),2.0,a);	
//		circle( this->img,ret[i],2.0, Scalar( 255, 255, 255 ),-1, 8 );
//		cout<<ret[i].x<<' '<<ret[i].y<<endl;
//	}
//	namedWindow( "Display window", WINDOW_NORMAL );
 // imshow( "Display window",this-> img );
//	waitKey(0);
	return ret;
}

*/
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }
		  
		  Faces[0].img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);                                 // Take in all the images, neutral1
		  Faces[1].img = cv::imread(argv[3],CV_LOAD_IMAGE_COLOR);											// Smile1
		  Faces[2].img = cv::imread(argv[4],CV_LOAD_IMAGE_COLOR);                                 // Neutral2
		  Faces[3].img = Mat::zeros(Faces[2].img.rows, Faces[2].img.cols, CV_8UC3);               // Tobsmile2
		  
		  //This initialises shape predictor sp,detector
		  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		  dlib::shape_predictor sp;
		  dlib::deserialize(argv[1]) >> sp;

		  dlib::image_window win, win_faces;

       for (int i = 2; i <= 4; ++i)
        {
            cout <<"processing image " << argv[i] << endl;      
				dlib::array2d<dlib::rgb_pixel> img;                                
				dlib::load_image(img, argv[i]);
				std::vector<dlib::rectangle> dets = detector(img);                  // dets bounds the faces with rectangles
																										
				if(dets.size()!= 1){
					cout<<"Error : No of faces detected in "<< argv[i] <<" != 1 !"<<endl;
					return 0;
				}
            std::vector<dlib::full_object_detection> shapes;
	         for (unsigned long j = 0; j < dets.size(); ++j)
            {
					dlib::full_object_detection shape = sp(img, dets[j]);
					Faces[i-2].shape = shape;
     //               cout << "number of parts: "<< shape.num_parts() << endl;
       //             cout << "pixel position of first part:  " << shape.part(1)<<endl;//.x()<<' '<<shape.part(0).y() << endl;
        //            cout << "pixel position of second part: " << shape.part(2) << endl;
					 for(int i =0; i<68; i++){
						 
						 dlib::rgb_pixel a;
						 dlib::draw_solid_circle ( img,shape.part(i),2.0,a);
//						 cv::Mat A= dlib::toMat (img);
//						 imshow("Points are",A);
//						 waitKey(500);
					 }
	
                shapes.push_back(shape);
            }
				Faces[i-2].equalize();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
//	 Faces[3].shape2 = Faces[3].Create_morphed_landMark(Faces[0].shape, Faces[1].shape, Faces[2].shape);
//	 cout<<"Landmarkpts created ....."<<endl;
	
	 Faces[0].DelaunayTriangulate(0);
//	 Faces[2]. DelaunayTriangulate(0);
	 cout<<"src triangulated...."<<endl;
	 
//	 cv::imshow("output", Faces[2].img);
//	 cv::waitKey(0);
//	 Faces[3]. DelaunayTriangulate(1);
	 cout<<"dest triangulated..."<<endl;
   
//	 for(int i =0; i< 150; i++){
//		 cout<<ind1[i]<<' ';
//	 }
//	 cout<<endl;


	// Faces[3].AffineTransform(Faces[2],Faces[3]);
}

// ----------------------------------------------------------------------------------------

