#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

const int m = 500; // theta resolution
const double PI = 3.14159265358979323846;

int params[4] = {0}; /*   0 - x coordinate of center
                          1 - y coordinate of center
                          2 - inner radius
                          3 - outer radius   */

string windowName = "Frame_1";

// linspace
vector<double> linspaceFunc(double a, double b, size_t N)
{
    double h = (b - a) / static_cast<double>(N - 1);
    vector<double> linspace(N);
    vector<double>::iterator x;
    double val;

    for (x = linspace.begin(), val = a; x != linspace.end(); ++x, val += h)
    {
        *x = val;
    }
    return linspace;
}

// range
vector<int> rangeFunc(int a, int b)
{
    int N = (b - a);
    vector<int> range(N);
    vector<int>::iterator x;
    double val;

    for (x = range.begin(), val = a; x != range.end(); ++x, val += 1)
    {
        *x = val;
    }
    return range;
}

// mean
int mean(vector<int> data)
{
    vector<int>::iterator x;
    int sum = 0;
    for (x = data.begin(); x != data.end(); ++x)
    {
        sum += *x;
    }
    return (int)(sum / data.size());
}

// deviation
double deviation(vector<int> radius_vec)
{
    size_t N = radius_vec.size();
    int sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += pow(radius_vec[i] - mean(radius_vec), 2);
    }
    return sqrt(sum / ((double)N));
}

// click event handler
void click_event(int event, int x, int y, int flags, void *data)
{
    // create copy of image after every click
    string *filename = (string *)data;
    Mat imageCopy = imread(*filename);

    if (event == EVENT_LBUTTONDOWN)
    {
        // store coordinate of click
        params[0] = x;
        params[1] = y;

        // show click coordinate on copy of image
        putText(imageCopy, to_string(x) + "," + to_string(y), Point(x - 25, y), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
        imshow(windowName, imageCopy);
    }
}

// slider handler
void on_change(int val, void *data)
{
    // create copy of image after every click
    string *filename = (string *)data;
    Mat imageCopy = imread(*filename);

    // getting slider values
    int radius_inner = getTrackbarPos("inner radius", windowName);
    int radius_outer = getTrackbarPos("outer radius", windowName);

    // store slide values
    if (radius_inner != 0)
        params[2] = radius_inner;
    if (radius_outer != 0 && radius_outer > radius_inner)
        params[3] = radius_outer;
    if (radius_outer < radius_inner)
        putText(imageCopy, "outer radius is smaller than inner radius", Point(params[0], params[1]), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    imshow(windowName, imageCopy);

    // displaying circles on the image
    circle(imageCopy, Point(params[0], params[1]), radius_inner, CV_RGB(0, 255, 0));
    circle(imageCopy, Point(params[0], params[1]), radius_outer, CV_RGB(255, 0, 0));
    imshow(windowName, imageCopy);
}

// contour detection
void contour_detection(Mat *img, vector<int> &contour_X, vector<int> &contour_Y, vector<int> &radius_vec, int mode)
{
    Mat arr_data = *img;
    vector<double> theta = linspaceFunc(0, 2 * PI, m);
    vector<int> radius = rangeFunc(params[2], params[3]);
    size_t n = radius.size();

    int x_c = params[0];
    int y_c = params[1];

    int polar_img[m][n] = {0};

    // cartesian to polar information
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int x = (int)(radius[j] * cos(theta[i]) + x_c);
            int y = (int)(radius[j] * sin(theta[i]) + y_c);

            if (x >= arr_data.cols - 1)
                break;
            if (y >= arr_data.rows - 1)
                break;
            polar_img[i][j] = arr_data.at<uchar>(y, x);
        }
    }

    // peak detection
    for (int i = 0; i < m; i++)
    {
        int min = 255, r;
        for (int j = 0; j < n; j++)
        {
            if (polar_img[i][j] <= min)
            {
                min = polar_img[i][j];
                r = radius[j];
            }
        }
        int x = (int)(r * cos(theta[i]) + x_c);
        int y = (int)(r * sin(theta[i]) + y_c);

        int t = (y % 2);
        if (mode == 0 & t == 0)

        {
            radius_vec.push_back(r);
            contour_X.push_back(x);
            contour_Y.push_back(y);
        }
        if (mode == 1 & t == 1)

        {
            radius_vec.push_back(r);
            contour_X.push_back(x);
            contour_Y.push_back(y);
        }
    }
}

// preprocessing
Mat preprocessing(Mat *img)
{
    Mat imageCopy = *img;
    equalizeHist(imageCopy, imageCopy);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(5);
    clahe->apply(imageCopy, imageCopy);
    blur(imageCopy, imageCopy, Size(5, 5), Point(-1, -1));
    GaussianBlur(imageCopy, imageCopy, Size(5, 5), 0, 0);
    return imageCopy;
}

// user input
void user_input(Mat img, string filename)
{
    // Create a window
    namedWindow(windowName, 1);

    // set the callback function for any mouse event
    setMouseCallback(windowName, click_event, &filename);

    // show the image
    imshow(windowName, img);

    if (filename == "/home/user/Sayar/Fluctuations/data/tripta_data/1.bmp")
    { // setting slider and calling slider value change handler function
        createTrackbar(
            "inner radius", windowName, 0, min(img.rows, img.cols), on_change, &filename);
        createTrackbar(
            "outer radius", windowName, 0, min(img.rows, img.cols), on_change, &filename);
    }

    // Wait until user press some key
    waitKey(0);

    // close the window
    destroyAllWindows();
}

// Driver code
int main(int argc, char **argv)
{
    cout << "Enter final frame number in integer: " << endl;
    int final_frame;
    cin >> final_frame;
    
    // Read image from file
    string filename = "/home/user/Sayar/Fluctuations/data/tripta_data/1.bmp";
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    user_input(img, filename);

    size_t frame_count = 0;

    // file to store data
    ofstream myfile;
    myfile.open("fourier_amplitudes.txt");
    myfile << "Each row stores amplitude data for each frame." << endl;
    myfile.close();
    myfile.open("legendre_amplitudes.txt");
    myfile << "Each row stores amplitude data for each frame." << endl;
    myfile.close();
    myfile.open("epsilon.txt");
    myfile << "Each row stores epsilon data for each frame." << endl;
    myfile.close();

    for (int i = 1; i < final_frame + 1; i++)
    {
        filename = "/home/user/Sayar/Fluctuations/data/tripta_data/1.bmp";

        size_t start_pos = filename.find("1");
        filename.replace(start_pos, 1, to_string(i));
        Mat img = imread(filename, IMREAD_GRAYSCALE);

        // if fail to read the image
        if (img.empty())
        {
            cout << "Error loading the image " << to_string(i) << ".bmp" << endl;
            i++;
        }
        else
        {
            img = preprocessing(&img);
            for (int mode = 0; mode < 2; mode++)
            {
                windowName = "Frame_1";
                vector<int> contour_X;
                vector<int> contour_Y;
                vector<int> radius_vec;

                contour_detection(&img, contour_X, contour_Y, radius_vec, mode);
                params[0] = mean(contour_X);
                params[1] = mean(contour_Y);

                // rejecting stage movement
                if (deviation(radius_vec) > 7)
                {
                    size_t start_pos = filename.find(to_string(i));
                    filename.replace(start_pos, to_string(i).length(), to_string(i + 1));
                    Mat img = imread(filename, IMREAD_GRAYSCALE);
                    i++;

                    if (!img.empty())
                    {
                        size_t start_pos = windowName.find("1");
                        windowName.replace(start_pos, 1, to_string(i));
                        user_input(img, filename);
                    }
                    else
                    {
                        while (true)
                        {
                            cout << "Error loading the image " << to_string(i) << ".bmp" << endl;
                            size_t start_pos = filename.find(to_string(i));
                            filename.replace(start_pos, to_string(i).length(), to_string(i + 1));
                            Mat img = imread(filename, IMREAD_GRAYSCALE);
                            i++;
                            if (!img.empty())
                            {
                                size_t start_pos = windowName.find("1");
                                windowName.replace(start_pos, 1, to_string(i));
                                user_input(img, filename);
                                break;
                            }
                            if (i > final_frame)
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {

                    for (int j = 0; j < contour_X.size(); j++)
                    {
                        circle(img, Point(contour_X[j], contour_Y[j]), 0, CV_RGB(255, 255, 255), -1);
                    }
                }

                // Autocorrelation
                int n = radius_vec.size();
                int R[n][n] = {0};

                for (int j = 0; j < n; j++)
                {
                    R[0][j] = radius_vec[j];
                }
                for (int j = 0; j < n; j++)
                {
                    int t = radius_vec[0];
                    radius_vec.erase(radius_vec.begin());
                    radius_vec.push_back(t);

                    for (int k = 0; k < n; k++)
                    {
                        R[j][k] = radius_vec[k];
                    }
                }

                // average radius
                double r_avg = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j % 2 == 0)
                    {
                        r_avg += 2 * radius_vec[j];
                    }
                    else
                    {
                        r_avg += 4 * radius_vec[j];
                    }
                }
                r_avg /= 3 * (double)n;

                vector<double> epsilon;
                for (int j = 0; j < n; j++)
                {
                    double t = 0;
                    for (int k = 0; k < n; k++)
                    {
                        if (k % 2 == 0)
                        {
                            t += 2 * (R[j][k] - r_avg) * (radius_vec[k] - r_avg);
                        }
                        else
                        {
                            t += 4 * (R[j][k] - r_avg) * (radius_vec[k] - r_avg);
                        }
                    }
                    epsilon.push_back(t / (double)(3 * n * pow(r_avg, 2)));
                }

                // storing autocorrelation
                myfile.open("epsilon.txt", ios_base::app);

                for (int j = 0; j < n; j++)
                {
                    myfile << to_string(epsilon[j]) + ",";
                }
                myfile << "\n";
                myfile.close();

                // Decomposition
                int k_max = 11;
                int l_max = 11;

                double a[k_max] = {0};
                double b_prime[l_max] = {0};
                double b[l_max] = {0};

                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < k_max; k++)
                    {
                        a[k] += cos(k * j * 2 * PI / (double)n) * epsilon[j];
                    }

                    b_prime[0] = abs(sin(j * 2 * PI / (double)n)) * epsilon[j];
                    b_prime[1] = cos(j * 2 * PI / (double)n) * b_prime[0];

                    for (int l = 2; l < l_max; l++)
                    {
                        b_prime[l] = (b_prime[l - 1] * cos(i * 2 * PI / n) * (2 * l - 1) - b_prime[l - 2] * (l - 1)) / l;
                    }

                    for (int l = 0; l < l_max; l++)
                    {
                        b[l] += (2 - j % 2) * b_prime[l];
                    }
                }
                for (int k = 0; k < k_max; k++)
                {
                    a[k] *= 2 / (double)n;
                }
                for (int l = 0; l < l_max; l++)
                {
                    b[l] *= PI * (2 * l + 1) / (3 * (double)n);
                }

                // storing amplitudes
                myfile.open("fourier_amplitudes.txt", ios_base::app);

                for (int k = 0; k < k_max; k++)
                {
                    myfile << to_string(a[k]) + ",";
                }

                myfile << "\n";
                myfile.close();

                myfile.open("legendre_amplitudes.txt", ios_base::app);

                for (int l = 0; l < l_max; l++)
                {
                    myfile << to_string(b[l]) + ",";
                }
                myfile << "\n";
                myfile.close();
            }

            string savelocation = "/home/user/Sayar/Fluctuations/Contour_new/";
            bool check = imwrite(savelocation + to_string(i) + ".bmp", img);

            // if the image is not saved
            if (check == false)
            {
                cout << "Saving the image " + to_string(i) + " FAILED" << endl;
            }
            else
            {
                cout << "Saving the image " + to_string(i) + " SUCCESSFUL" << endl;
                frame_count++;
            }
        }
    }
    cout << "frame count is " << frame_count << endl;
    return 0;

    // todo: outlier removal

    // todo: GPU optimization
    // todo: split code
}