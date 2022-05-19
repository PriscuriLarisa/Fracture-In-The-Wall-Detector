// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <math.h>

std::vector<int> generateHistogram(Mat_<uchar> src) {
	int height = src.rows;
	int width = src.cols;
	std::vector<int> Ng(256, 0);

	//create histogram
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Ng[src(i, j)]++;
		}
	}

	return Ng;
}

std::vector<int> generateSpecificHistogram(Mat_<uchar> src, Mat_<uchar> labels) {
	int height = src.rows;
	int width = src.cols;
	std::vector<int> Ng(256, 0);

	//create histogram
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (labels(i, j) == 255)
				Ng[src(i, j)]++;
		}
	}

	return Ng;
}

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}



Mat_<uchar> bin(Mat_<uchar> src, int trashold) {
	Mat_<uchar> dst = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) > trashold)
				dst(i, j) = 255;
			else
				dst(i, j) = 0;
		}
	}

	return dst;
}

int labeling(Mat_<uchar> src, Mat labels) {
	
	int numberOfLabels = connectedComponents(src, labels, 8);

	return numberOfLabels;
}

Mat_<uchar> adaptiveBinarize(Mat_<uchar> src, Mat_<uchar> labels) {
	std::vector<int> histogram = generateSpecificHistogram(src, labels);
	int h[256];

	Mat_<uchar> dst = src.clone();
	int Imax = -1, Imin = -1;
	for (int i = 0; i < 256 && (Imin == -1 || Imax == -1); i++) {
		if (Imin == -1 && histogram[i] != 0)
			Imin = i;
		if (Imax == -1 && histogram[255 - i] != 0)
			Imax = 255 - i;
	}

	int Tk = (Imin + Imax) / 2;
	int Tk1 = (Imin + Imax) / 2;

	do {
		int N1 = 0, N2 = 0;
		int nG1 = 0, nG2 = 0;

		for (int g = Imin; g <= Tk1; g++) {
			N1 += histogram[g];
			nG1 += g * histogram[g];
		}

		for (int g = Tk1; g <= Imax; g++) {
			N2 += histogram[g];
			nG2 += g * histogram[g];
		}

		nG1 /= N1;
		nG2 /= N2;

		Tk = Tk1;
		Tk1 = (nG1 + nG2) / 2;

		if (Tk1 < 0)
			Tk1 = 0;
		else if (Tk1 > 255)
			Tk1 = 255;

	} while (abs(Tk1 - Tk) > 0.1);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (src(i, j) < Tk1)
				dst(i, j) = 255;
			else
				dst(i, j) = 0;
		}
	}

	return dst;
}

Mat_<uchar> specificAdaptiveBinarize(Mat_<uchar> src, Mat_<uchar> labels) {
	std::vector<int> histogram = generateSpecificHistogram(src, labels);
	int h[256];

	Mat_<uchar> dst = src.clone();
	int Imax = -1, TH = -1;
	for (int i = 0; i < 256; i++) {
		if (histogram[i] > Imax) {
			Imax = histogram[i];
			TH = i;
		}

	}

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (src(i, j) <= TH)
				dst(i, j) = 255;
			else
				dst(i, j) = 0;
		}
	}


	return dst;
}

float calculateElongation(Mat labels, int label) {
	int maxY = -1, maxX = -1, xMin=-1, yMin=-1;
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (xMin == -1 && labels.at<int>(i, j) == label)
				xMin = i;
			if (yMin == -1 && labels.at<int>(i, j) == label)
				yMin = j;
			if (labels.at<int>(i, j) == label && i > maxX)
				maxX = i;
			if (labels.at<int>(i, j) == label && j > maxY)
				maxY = j;
		}
	}
	
	return (float)(maxX- xMin +1) / (maxY- yMin+1);
}

std::tuple<std::vector<int>, float> filterObjects(Mat labels, int nbOfLabels, float TH) {
	float max1=-1, max2=-1;
	std::vector<int> filteredLabels(nbOfLabels, 0);

	for (int i = 1; i < nbOfLabels; i++) {
		float elongation = calculateElongation(labels, i);
		if ((elongation > TH)) {
			filteredLabels[i] = i;
		}
		else if ((float)1 / elongation < (float)1 / TH) {
			filteredLabels[i] = i;
		}
		else
			filteredLabels[i] = 0;
		if (elongation > max1) {
			max2 = max1;
			max1 = elongation;
		}
		else if (elongation > max2) {
			max2 = elongation;
		}
	}
	return std::make_tuple(filteredLabels, max2);
}

Mat_<uchar> reconstructImage(Mat labels, Mat_<uchar> src, std::vector<int> filteredObjects) {
	Mat_<uchar> dst = src.clone();
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if(labels.at<int>(i, j)< filteredObjects.size()-1)
				if (filteredObjects[labels.at<int>(i, j)] != 0) {
					dst[i][j] = 255;
				}
				else
					dst[i][j] = 0;
		}
	}
	return dst;
}



Mat applyBlackHat(Mat src, Mat dst, Mat element) {
	morphologyEx(src, dst, MORPH_BLACKHAT, element);
	return dst;
}

Mat applyOpening(Mat src, Mat dst, Mat element) {
	morphologyEx(src, dst, MORPH_OPEN, element);

	return dst;
}

Mat applyDilate(Mat src, Mat dst, Mat element) {
	morphologyEx(src, dst, MORPH_DILATE, element);	
	return dst;
}


std::tuple<Mat, float> applyOperations(Mat_<uchar> src, Mat_<uchar> binarized, float elongation, int iteration) {
	Mat element, dstDilate, dstOpened2;
	int step = 0;

	element = getStructuringElement(2, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
	dstDilate = applyDilate(binarized, dstDilate, element);

	char windowName[100];
	sprintf(windowName, "%d %s %d %s", iteration, ".", step, " Dilatare");
	imshow(windowName, dstDilate);
	step++;

	if (iteration == 1) {
		element = getStructuringElement(2, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
		dstOpened2 = applyOpening(dstDilate, dstOpened2, element);
		sprintf(windowName, "%d %s %d %s", iteration, ".", step, " Deschidere");
		imshow(windowName, dstOpened2);
		step++;
	}
	else
		dstOpened2 = dstDilate.clone();


	Mat labels(src.size(), CV_32S);
	int nbOfLabels = labeling(dstOpened2, labels);
	auto filteredObjects = filterObjects(labels, nbOfLabels, elongation);
	std::vector<int> filteredLabels = std::get<0>(filteredObjects);
	elongation = std::get<1>(filteredObjects);

	Mat_<uchar> filteredImage = reconstructImage(labels, dstOpened2, filteredLabels);
	sprintf(windowName, "%d %s %d %s", iteration, ".", step, " Filtrare");
	imshow(windowName, filteredImage);
	step++;


	Mat_<uchar> dstBinarized2 = adaptiveBinarize(src, filteredImage);
	sprintf(windowName, "%d %s %d %s", iteration, ".", step, " Binarizare Adaptativa");
	imshow(windowName, dstBinarized2);
	step++;

	return std::make_tuple(dstBinarized2, elongation);
}

Mat_<uchar> applyInitialOperations(Mat_<uchar> src, int morph_size, int trashold) {

	Mat dstEqualize, dstBlackHat, dstTopHat, dstOpened, dstBinarizedAdaptive, dstDilate;
	equalizeHist(src, dstEqualize);

	imshow("0.1. Histograma egalizata", dstEqualize);
	Mat element;

	element = getStructuringElement(2, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	dstBlackHat = applyBlackHat(dstEqualize, dstBlackHat, element);

	imshow("0.2. Aplicare BlackHat", dstBlackHat);

	Mat_<uchar> dstBinarized = bin(dstBlackHat, trashold);
	imshow("0.3. Binarizare hardcodata", dstBinarized);

	return dstBinarized;
}

void crack()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		int morph_elem = 0;
		int morph_size = 30;
		int morph_operator = 0;
		int const max_operator = 4;
		int const max_elem = 2;
		int const max_kernel_size = 21;
		int trashold = 140;
		int iteration = 1;

		Mat_<uchar> src;
		Mat dstEqualize, dstBlackHat, dstTopHat, dstOpened, dstBinarizedAdaptive, dstDilate;
		Mat_<uchar> dstOpened2;
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("0.0. Source", src);

		Mat_<uchar> dstBinarized = applyInitialOperations(src, morph_size, trashold);

		char windowName[100];
		
		auto complexObject = applyOperations(src, dstBinarized, 15.0f, iteration);
		Mat resultMat = std::get<0>(complexObject);
		sprintf(windowName, "%d %s", iteration, "-> Rezultat Binar");
		imshow(windowName, resultMat);
		iteration++;
		float elongation = std::get<1>(complexObject);

		complexObject = applyOperations(src, resultMat, elongation, iteration);
		Mat resultMat2 = std::get<0>(complexObject);
		sprintf(windowName, "%d %s", iteration,"-> Rezultat Binar");
		imshow(windowName, resultMat2);
		iteration++;
		elongation = std::get<1>(complexObject);

		complexObject = applyOperations(src, resultMat2, elongation, iteration);
		resultMat2 = std::get<0>(complexObject);
		sprintf(windowName, "%d %s", iteration, "-> Rezultat Binar");
		imshow(windowName, resultMat2);
		iteration++;
		elongation = std::get<1>(complexObject);

		complexObject = applyOperations(src, resultMat2, elongation, iteration);
		resultMat2 = std::get<0>(complexObject);
		sprintf(windowName, "%d %s", iteration, "-> Rezultat Binar");
		imshow(windowName, resultMat2);
		iteration++;
		elongation = std::get<1>(complexObject);

		complexObject = applyOperations(src, resultMat2, elongation, iteration);
		resultMat2 = std::get<0>(complexObject);
		sprintf(windowName, "%d %s", iteration, "-> Rezultat Binar");
		imshow(windowName, resultMat2);
		iteration++;
		elongation = std::get<1>(complexObject);

		waitKey();

	}
}


int main()
{
	int op;
	int save, it, r;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Crack\n\n");

		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			crack();
			break;

		}
	} while (op != 0);
	return 0;
}