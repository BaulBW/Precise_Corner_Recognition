// precise corner recognition of rounding corners with uncertainty smaller than 3 px
// Recognize corners as the intersections of two perpendicular lines

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// class for coordinate
class coord {
public:
	float coordx;
	float coordy;
	coord() : coordx(0), coordy(0) {};
	coord(float x, float y) : coordx(x), coordy(y) {};
	coord plus(coord dt) {
		return coord(coordx + dt.coordx, coordy + dt.coordy);
	}
	coord times(int m, int n) {
		return coord(coordx * m, coordy * n);
	}
	void printcd()
	{
		cout << "\n\nx: " << coordx << ", y: " << coordy << "\n\n";
	}
};


Mat src, src2
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

int main()
{
	src = imread("source.png", -1); // import as a gray image
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}
	
	GaussianBlur(src, src2, Size(5, 5), 0, 0); // blur
	adaptiveThreshold(src2, src2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 2); // Edge detection, Canny doesn't perform as well for the purpose here
	// Canny(src2, src2, 150, 300, 3);
	imwrite("Edges.png", src2);
	

	// show source image
	resize(src, src, Size(src.cols / 4, src.rows / 4));
	imshow(source_window, src);

	Mat cdstP = src.clone();

	int cnt = 0;
	double averagex = 0, averagey = 0;

	// Probabilistic Line Transform
	for (int j = 0; j < 2; ++j) {
		vector<Vec4i> linesP; // will hold the results of the detection
		HoughLinesP(src2, linesP, 1, CV_PI / 180, 50, j == 0 ? 100 : 50, 10); // runs Hough detection
		
		// Draw the lines
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i l = linesP[i];
			line(cdstP, Point(round(((double)l[0]) / 4), round(((double)l[1]) / 4)), Point(round(((double)l[2]) / 4), round(((double)l[3]) / 4)), Scalar(0, 0, 255), 1, LINE_AA);
		}

		// Corners are the intersection of two perpendicular lines
		for (size_t i = 0; i < linesP.size(); i++)
			for (size_t j = i + 1; j < linesP.size(); j++)
			{
				double a1 = linesP[i][0], b1 = linesP[i][1], c1 = linesP[i][2], d1 = linesP[i][3];
				double a2 = linesP[j][0], b2 = linesP[j][1], c2 = linesP[j][2], d2 = linesP[j][3];

				if (abs((c1 - a1) * (c2 - a2) + (d1 - b1) * (d2 - b2)) >= 0.01) // proceed only if the two lines are perpendicular, which is judged by inner product of vectors
				{
					continue;
				}

				double A1 = d1 - b1, B1 = a1 - c1, C1 = a1 * (d1 - b1) + b1 * (a1 - c1),
					A2 = d2 - b2, B2 = a2 - c2, C2 = a2 * (d2 - b2) + b2 * (a2 - c2);
				double denominator = A1 * B2 - B1 * A2;

				double x = (C1 * B2 - B1 * C2) / denominator, y = (A1 * C2 - C1 * A2) / denominator;
				averagex += x, averagey += y;
				cnt++;
			}

		// averaging only; classification may be necessary with different adaptiveThreshold paramters
		if (cnt > 0) 
		{
			Point pt;
			pt.x = cvRound(averagex / cnt / 4), pt.y = cvRound(averagey / cnt / 4);
			circle(cdstP, pt, 30, Scalar(0, 0, 255), FILLED, LINE_8);
			break;
		}
	}
	if (cnt == 0)
		cout << "\n\n" << "recognition failed!" << "\n\n";
	else
		coord(averagex / cnt, averagey / cnt).printcd();

	// Show results
	imshow(corners_window, cdstP);
	imwrite("Corners.png", cdstP);
	waitKey();

	return 0;
}
