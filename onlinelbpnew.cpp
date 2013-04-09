#include <stdio.h>
#if defined WIN32 || defined _WIN32
	#include <conio.h>		// For _kbhit() on Windows
	#include <direct.h>		// For mkdir(path) on Windows
	#define snprintf sprintf_s	// Visual Studio on Windows comes with sprintf_s() instead of snprintf()
#else
	#include <stdio.h>		// For getchar() on Linux
	#include <termios.h>	// For kbhit() on Linux
	#include <unistd.h>
	#include <sys/types.h>
	#include <sys/stat.h>	// For mkdir(path, options) on Linux
#endif
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
//#include <string.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifndef BOOL
	#define BOOL bool
#endif

using namespace std;
using namespace cv;

// Haar Cascade file, used for Face Detection.
const char *faceCascadeFilename = "haarcascade_frontalface_alt.xml";

int SAVE_EIGENFACE_IMAGES = 1;		// Set to 0 if you dont want images of the Eigenvectors saved to files (for debugging).
//#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.


// Global variables
IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
//#define	MAX_NAME_LENGTH 256		// Give each name a fixed size for easier code.
//char **personNames = 0;			// array of person names (indexed by the person number). Added by Shervin.
vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.
int faceWidth = 120;	// Default dimensions for faces in the face recognition database. Added by Shervin.
int faceHeight = 90;	//	"		"		"		"		"		"		"		"
int nPersons                  = 0; // the number of people in the training set. Added by Shervin.
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces

vector<Mat> images;
vector<int> labels;
int predicted_label = -1;
double predicted_confidence = 0.0;

Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(1,8,8,8,90.0);

//CvCapture* camera = 0;	// The camera device.
CvCapture* camera;
bool loaded=false;
// Function prototypes
void printUsage();
void recognizeFromCam(void);
Mat getCameraFrame(void);
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
void saveFloatImage(const char *filename, const IplImage *srcImg);
CvRect detectFaceInImage(const Mat inputMat, CascadeClassifier cascade );
bool retrainOnline(void);
bool loaddata();

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ' ') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, classlabel, separator);
        getline(liness, path,separator);
	getline(liness, path,separator);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

// Show how to use this program from the command-line.
void printUsage()
{
	printf("OnlineFaceRec, created by Shervin Emami (www.shervinemami.co.cc), 2nd Jun 2010.\n"
		"Usage: OnlineFaceRec [<command>] \n"
		"  Valid commands are: \n"
		"    train <train_file> \n"
		"    test <test_file> \n"
		" (if no args are supplied, then online camera mode is enabled).\n"
		);
}


// Startup routine.
int main( int argc, char** argv )
{
	printUsage();

	if( argc >= 2 && strcmp(argv[1], "train") == 0 ) {
		char *szFileTrain;
		if (argc == 3)
			szFileTrain = argv[2];	// use the given arg
		else {
			printf("ERROR: No training file given.\n");
			return 1;
		}
		//learn(szFileTrain);
	}
	else if( argc >= 2 && strcmp(argv[1], "test") == 0) {
		char *szFileTest;
		if (argc == 3)
			szFileTest = argv[2];	// use the given arg
		else {
			printf("ERROR: No testing file given.\n");
			return 1;
		}
		//recognizeFileList(szFileTest);
	}
	else {
		recognizeFromCam();
	}
	return 0;
}

#if defined WIN32 || defined _WIN32
	// Wrappers of kbhit() and getch() for Windows:
	#define changeKeyboardMode
	#define kbhit _kbhit
#else
	// Create an equivalent to kbhit() and getch() for Linux,
	// based on "http://cboard.cprogramming.com/c-programming/63166-kbhit-linux.html":
	
	#define VK_ESCAPE 0x1B		// Escape character

	// If 'dir' is 1, get the Linux terminal to return the 1st keypress instead of waiting for an ENTER key.
	// If 'dir' is 0, will reset the terminal back to the original settings.
	void changeKeyboardMode(int dir)
	{
		static struct termios oldt, newt;

		if ( dir == 1 ) {
			tcgetattr( STDIN_FILENO, &oldt);
			newt = oldt;
			newt.c_lflag &= ~( ICANON | ECHO );
			tcsetattr( STDIN_FILENO, TCSANOW, &newt);
		}
		else
			tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
	}

	// Get the next keypress.
	int kbhit(void)
	{
		struct timeval tv;
		fd_set rdfs;

		tv.tv_sec = 0;
		tv.tv_usec = 0;

		FD_ZERO(&rdfs);
		FD_SET (STDIN_FILENO, &rdfs);

		select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
		return FD_ISSET(STDIN_FILENO, &rdfs);
	}

	// Use getchar() on Linux instead of getch().
	#define getch() getchar()
#endif






// Grab the next camera frame. Waits until the next frame is ready,
// and provides direct access to it, so do NOT modify the returned image or free it!
// Will automatically initialize the camera on the first frame.
Mat getCameraFrame(void)
{
	Mat frame;

	// If the camera hasn't been initialized, then open it.
	if (!camera) {
		printf("Acessing the camera ...\n");
		camera = cvCaptureFromCAM( 0 );
		if (!camera) {
			printf("ERROR in getCameraFrame(): Couldn't access the camera.\n");
			exit(1);
		}
		// Try to set the camera resolution
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 320 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 240 );
		// Wait a little, so that the camera can auto-adjust itself
		#if defined WIN32 || defined _WIN32
			Sleep(1000);	// (in milliseconds)
		#endif
		frame = cvQueryFrame( camera );	// get the first frame, to make sure the camera is initialized.
		if (!frame.empty()) {
			printf("Got a camera using a resolution of %dx%d.\n", (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT) );
		}
	}

	frame = cvQueryFrame( camera );
	if (frame.empty()) {
		fprintf(stderr, "ERROR in recognizeFromCam(): Could not access the camera or video file.\n");
		exit(1);
		//return NULL;
	}
	return frame;
}

// Get an 8-bit equivalent of the 32-bit Float image.
// Returns a new image, so remember to call 'cvReleaseImage()' on the result.
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// Spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);

		//cout << "FloatImage:(minV=" << minVal << ", maxV=" << maxVal << ")." << endl;

		// Deal with NaN and extreme values, since the DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal-minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove potential divide by zero errors.

		// Convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal-minVal));
	}
	return dstImg;
}

// Store a greyscale floating-point CvMat image into a BMP/JPG/GIF/PNG image,
// since cvSaveImage() can only handle 8bit images (not 32bit float images).
void saveFloatImage(const char *filename, const IplImage *srcImg)
{
	//cout << "Saving Float Image '" << filename << "' (" << srcImg->width << "," << srcImg->height << "). " << endl;
	IplImage *byteImg = convertFloatImageToUcharImage(srcImg);
	cvSaveImage(filename, byteImg);
	cvReleaseImage(&byteImg);
}

// Perform face detection on the input image, using the given Haar cascade classifier.
// Returns a rectangle for the detected region in the given image.
CvRect detectFaceInImage(const Mat inputMat, CascadeClassifier cascade )
{
	const CvSize minFeatureSize = cvSize(50, 50);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;	// Only search for 1 face.
	const float search_scale_factor = 1.2f;
	//IplImage *detectImg;
	//IplImage *greyImg = 0;
	Mat detectMat,greyMat;
	//CvMemStorage* storage;
	CvRect rc,ry;
	double t;
	//CvSeq* rects;
	std::vector<Rect> rects;
	int i;

	
	// Detect all the faces.
	t = (double)cvGetTickCount();

	cascade.detectMultiScale(inputMat,rects,1.1,3,flags,minFeatureSize);	
	t = (double)cvGetTickCount() - t;
	printf("[Face Detection took %d ms and found %ld objects]\n", cvRound( t/((double)cvGetTickFrequency()*1000.0) ), rects.size() );
        
	// Get the first detected face (the biggest).
	if (rects.size() > 0) {
        ry=rects[0]; 
        rc.x=ry.x+(2*ry.width/15);
        rc.y=ry.y+(ry.height/5);
        rc.width=ry.width-(3*ry.width/12);
        rc.height=ry.height-(ry.height/5);
	
	 
       // rc = *(CvRect*)cvGetSeqElem( rects, 0 );
    }
	else
		rc = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	return rc;	// Return the biggest face found, or (-1,-1,-1,-1).
}

// Re-train the new face rec database without shutting down.
// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
bool retrainOnline(void)
{
	CvMat *trainPersonNumMat;
        // Get the path to your CSV.
    	string fn_csv = "train.txt";
    	// These vectors hold the images and corresponding labels.
    	// Read in the data. This can fail if no valid
    	// input filename is given.
    	try {
        	read_csv(fn_csv, images, labels);
    	} 
	catch (cv::Exception& e) {
        	cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
       	 	// nothing more we can do
        	return false;
    	}
	cout<<"read from csv"<<endl;
	vector<int>::const_iterator it, it2;  
  	// Find the min and max elements in the vector
  	it2 = max_element(labels.begin(), labels.end());
  	cout << " the number of persons : " << *it2 << endl;
	nPersons=*it2;	
	model->train(images, labels);
	cout<<"Training models completed"<<endl;
	model->save("facedata.yml");
	cout<<"Saved the model"<<endl;
	 // Here is how to get the eigenvalues of this Eigenfaces model:
    	Mat eigenvalues = model->getMat("eigenvalues");
    	// And we can do the same to display the Eigenvectors (read Eigenfaces):
    	Mat W = model->getMat("eigenvectors");
    	// Get the sample mean from the training data
   	 Mat mean = model->getMat("mean");
	imwrite("mean.png",mean.reshape(1, images[0].rows));
	loaded=true;
	return true;
}

bool loadData(void)
{
        // Get the path to your CSV.
    	string fn_csv = "train.txt";
    	// These vectors hold the images and corresponding labels.
    	// Read in the data. This can fail if no valid
    	// input filename is given.
    	try {
        	read_csv(fn_csv, images, labels);
    	} 
	catch (cv::Exception& e) {
        	cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
       	 	// nothing more we can do
        	return false;
    	}
	cout<<"read from csv"<<endl;
	vector<int>::const_iterator it, it2;  
  	// Find the min and max elements in the vector
  	it2 = max_element(labels.begin(), labels.end());
  	cout << " the number of persons : " << *it2 << endl;	
	nPersons=*it2;
	CvFileStorage * fileStorage;
	fileStorage = cvOpenFileStorage( "facedata.yml", 0, CV_STORAGE_READ );
	if( !fileStorage ) {
		printf("Can't open training database file 'facedata.yml'.\n");
		loaded=false;
		return false;
	}
	else
	{
	   
	   model->load(fileStorage);
	   loaded=true;
	   cout<<"Loaded : Number of persons :"<<nPersons<<endl;
	}
	
	return true;
}

// Continuously recognize the person in the camera.
void recognizeFromCam(void)
{
	int i;
	CvMat * trainPersonNumMat;  // the person numbers during training
	float * projectedTestFace;//loadTrainingData
	double timeFaceRecognizeStart;
	double tallyFaceRecognizeTime;
        CascadeClassifier faceCascade;
	char cstr[256];
	BOOL saveNextFaces = FALSE;
	char newPersonName[256];
	int newPersonFaces;

	trainPersonNumMat = 0;  // the person numbers during training
	projectedTestFace = 0;
	saveNextFaces = FALSE;
	newPersonFaces = 0;

	printf("Recognizing person in the camera ...\n");

	loadData();
	// Project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

	// Create a GUI window for the user to see the camera image.
	cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);

	// Make sure there is a "data" folder, for storing the new person.
	#if defined WIN32 || defined _WIN32
		mkdir("data");
	#else
		// For Linux, make the folder to be Read-Write-Executable for this user & group but only Readable for others.
		mkdir("data", S_IRWXU | S_IRWXG | S_IROTH);
	#endif

	// Load the HaarCascade classifier for face detection.
	//faceCascade.load(faceCascadeFilename);
	if( !faceCascade.load(faceCascadeFilename) ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", faceCascadeFilename);
		exit(1);
	}

	// Tell the Linux terminal to return the 1st keypress instead of waiting for an ENTER key.
	changeKeyboardMode(1);

	timeFaceRecognizeStart = (double)cvGetTickCount();	// Record the timing.

	while (1)
	{
		int iNearest, nearest, truth;
		//IplImage *camImg;
		Mat camMat;
		Mat greyMat;
		Mat faceMat;
		Mat sizedMat;
		Mat equalizedMat;
		Mat processedFaceMat;
		Mat shownMat;
		/*IplImage *greyImg;
		IplImage *faceImg;
		IplImage *sizedImg;
		IplImage *equalizedImg;
		IplImage *processedFaceImg;*/
		CvRect faceRect;
//		IplImage *shownImg;
		int keyPressed = 0;
		FILE *trainFile;
		float confidence;
		
		// Handle non-blocking keyboard input in the console.
		if (kbhit())
			keyPressed = getch();
		
		if (keyPressed == VK_ESCAPE) {	// Check if the user hit the 'Escape' key
			break;	// Stop processing input.
		}
		switch (keyPressed) {
			case 'n':	// Add a new person to the training set.
				// Train from the following images.
				
				printf("Enter your name: ");
				strcpy(newPersonName, "newPerson");

				// Read a string from the console. Waits until they hit ENTER.
				changeKeyboardMode(0);
				loaded=false;				
				fgets(newPersonName, sizeof(newPersonName)-1, stdin);
				changeKeyboardMode(1);
				// Remove 1 or 2 newline characters if they were appended (eg: Linux).
				i = strlen(newPersonName);
				if (i > 0 && (newPersonName[i-1] == 10 || newPersonName[i-1] == 13)) {
					newPersonName[i-1] = 0;
					i--;
				}
				if (i > 0 && (newPersonName[i-1] == 10 || newPersonName[i-1] == 13)) {
					newPersonName[i-1] = 0;
					i--;
				}
				
				if (i > 0) {
					printf("Collecting all images until you hit 't', to start Training the images as '%s' ...\n", newPersonName);
					newPersonFaces = 0;	// restart training a new person
					saveNextFaces = TRUE;
				}
				else {
					printf("Did not get a valid name from you, so will ignore it. Hit 'n' to retry.\n");
				}
				break;
			case 't':	// Start training
				
				saveNextFaces = FALSE;	// stop saving next faces.
				// Store the saved data into the training file.
				printf("Storing the training data for new person '%s'.\n", newPersonName);
				// Append the new person to the end of the training data.
				trainFile = fopen("train.txt", "a");
				for (i=0; i<newPersonFaces; i++) {
					snprintf(cstr, sizeof(cstr)-1, "data/%d_%s%d.pgm", nPersons+1, newPersonName, i+1);
					fprintf(trainFile, "%d %s %s\n", nPersons+1, newPersonName, cstr);
				}
				fclose(trainFile);

				// Now there is one more person in the database, ready for retraining.
				nPersons++;

				break;
			case 'r':
				
				// Re-initialize the local data.
				projectedTestFace = 0;
				saveNextFaces = FALSE;
				newPersonFaces = 0;

				// Retrain from the new database without shutting down.
				// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
				cvFree( &trainPersonNumMat );	// Free the previous data before getting new data
				retrainOnline();
				// Project the test images onto the PCA subspace
				cvFree(&projectedTestFace);	// Free the previous data before getting new data
				//projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

				printf("Recognizing person in the camera ...\n");
				continue;	// Begin with the next frame.
				break;
		}

		// Get the camera frame
		// camImg = getCameraFrame();
		camMat = getCameraFrame();
		if (camMat.empty()) {
			printf("ERROR in recognizeFromCam(): Bad input image!\n");
			exit(1);
		}
		// Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
		//greyImg = convertImageToGreyscale(camImg);
		cvtColor( camMat, greyMat, CV_BGR2GRAY );
		// Perform face detection on the input image, using the given Haar cascade classifier.
		faceRect = detectFaceInImage(greyMat, faceCascade );

		// Make sure a valid face was detected.
		if (faceRect.width > 0) {
			//faceMat = cropImage(greyMat, faceRect);	// Get the detected face image.
			faceMat = greyMat(faceRect);			
			// Make sure the image is the same dimensions as the training images.
  			cv::resize(faceMat, sizedMat, Size(faceWidth,faceHeight));
			equalizeHist( sizedMat, equalizedMat );
			processedFaceMat=equalizedMat;
			cout<<"loaded : "<<loaded<<endl;
			if(loaded)
			{
				model->predict(processedFaceMat, predicted_label, predicted_confidence);
				cout<<"predictedLabel is : "<<predicted_label<<endl;
			}
			

			// Possibly save the processed face to the training set.
			if (saveNextFaces) {
// MAYBE GET IT TO ONLY TRAIN SOME IMAGES ?
				// Use a different filename each time.
				snprintf(cstr, sizeof(cstr)-1, "data/%d_%s%d.pgm", nPersons+1, newPersonName, newPersonFaces+1);
				printf("Storing the current face of '%s' into image '%s'.\n", newPersonName, cstr);
				//cvSaveImage(cstr, processedFaceImg, NULL);
				imwrite(cstr,processedFaceMat);
				newPersonFaces++;
                                       if(newPersonFaces>5)
                                        {
                                        saveNextFaces = FALSE;
                                        }
			}
			
			
		}

		// Show the data on the screen.
		shownMat = camMat;
		if (faceRect.width > 0) {	// Check if a face was detected.
			// Show the detected face region.
			rectangle(shownMat, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);
			if (predicted_label > 0 && loaded && predicted_confidence <90.0) {	// Check if the face recognition database is loaded and a person was recognized.
				// Show the name of the recognized person, overlayed on the image below their face.
				
				Scalar textColor = CV_RGB(0,255,255);	// light blue text
				char text[256];
				snprintf(text, sizeof(text)-1, "Name: '%d'", predicted_label);
				putText(shownMat, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), FONT_HERSHEY_COMPLEX_SMALL,0.5, textColor);
				snprintf(text, sizeof(text)-1, "Confidence: %f", predicted_confidence);
				putText(shownMat, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 30), FONT_HERSHEY_COMPLEX_SMALL, 0.5, textColor);
			}
		}

		// Display the image.
		//cvShowImage("Input", shownImg);
		imshow("Input",shownMat);
		// Give some time for OpenCV to draw the GUI and check if the user has pressed something in the GUI window.
		keyPressed = cvWaitKey(10);
		if (keyPressed == VK_ESCAPE) {	// Check if the user hit the 'Escape' key in the GUI window.
			break;	// Stop processing input.
		}

		//cvReleaseImage( &shownImg );
	}
	tallyFaceRecognizeTime = (double)cvGetTickCount() - timeFaceRecognizeStart;

	// Reset the Linux terminal back to the original settings.
	changeKeyboardMode(0);

	// Free the camera and memory resources used.
	//cvReleaseCapture( &camera );
	//cvReleaseHaarClassifierCascade( &faceCascade );
}
