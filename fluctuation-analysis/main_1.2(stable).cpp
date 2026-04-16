// Non-interlacing and angular scan

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

const double PI = M_PI; // PI

int params[4] = {0}; /*   0 - x coordinate of center
                          1 - y coordinate of center
                          2 - inner radius
                          3 - outer radius   */

string windowName = "Frame";

// linspace
vector<double> linspaceFunc(double a, double b, size_t N)
{
    double h = (b - a) / (double)(N - 1);
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
    size_t N = (b - a);
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
template <typename T>
T mean(vector<T> data)
{
    size_t N = data.size();
    T sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += data[i];
    }
    return (T)(sum / N);
}

// deviation
template <typename T>
double deviation(vector<T> radius_vec)
{
    size_t N = radius_vec.size();
    T sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += pow(radius_vec[i] - mean(radius_vec), 2);
    }
    return sqrt(sum / N);
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
        circle(imageCopy, Point(x, y), 2, CV_RGB(255, 255, 255), -1);

        // show circle with user given radii on copy of image
        circle(imageCopy, Point(x, y), params[2], CV_RGB(0, 255, 0));
        circle(imageCopy, Point(x, y), params[3], CV_RGB(255, 0, 0));
        imshow(windowName, imageCopy);
    }
}

// slider handler
void on_change(int val, void *data)
{
    // create copy of image after every click
    string *filename = (string *)data;
    Mat imageCopy = imread(*filename);

    // get slider values
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

    // show circle with user given radii on copy of image
    circle(imageCopy, Point(params[0], params[1]), radius_inner, CV_RGB(0, 255, 0));
    circle(imageCopy, Point(params[0], params[1]), radius_outer, CV_RGB(255, 0, 0));
    // show click coordinate on copy of image
    circle(imageCopy, Point(params[0], params[1]), 2, CV_RGB(255, 255, 255), -1);
    imshow(windowName, imageCopy);
}

// contour detection todo: gpu
void contour_detection(Mat *img, vector<int> &contour_X, vector<int> &contour_Y, vector<double> &radius_vec, size_t m, size_t n)
{
    // store image data as array
    Mat arr_data = *img;

    // initialize theta and radius vector
    vector<double> theta = linspaceFunc(0, 2 * PI, m);
    vector<double> radius = linspaceFunc(params[2], params[3], n);

    // initializing GUV center
    int x_c = params[0];
    int y_c = params[1];

    // initialize polar image
    int polar_img[m][n] = {0};

    // cartesian to polar information
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int x = (int)round((radius[j] * cos(theta[i]) + x_c));
            int y = (int)round((radius[j] * sin(theta[i]) + y_c));

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

        // store data
        radius_vec.push_back(r);
        int x = (int)round((r * cos(theta[i]) + x_c));
        int y = (int)round((r * sin(theta[i]) + y_c));
        contour_X.push_back(x);
        contour_Y.push_back(y);
    }
}

// filename padding
string padding(int s, int digits)
{
    string str = to_string(s);
    int precision = digits - min(digits, (int)str.size());
    string frame = string(precision, '0').append(str);
    return frame;
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
void user_input(Mat img, string filename, string first_frame_directory)
{
    // Create a window
    namedWindow(windowName, 1);

    // set the callback function for any mouse event
    setMouseCallback(windowName, click_event, &filename);

    // show the image
    imshow(windowName, img);

    if (filename == first_frame_directory)
    { // set slider and calling slider value change handler function
        createTrackbar(
            "inner radius", windowName, 0, min(img.rows, img.cols), on_change, &filename);
        createTrackbar(
            "outer radius", windowName, 0, min(img.rows, img.cols), on_change, &filename);
    }

    // wait until user press some key
    waitKey(0);

    // close the window
    destroyAllWindows();
}

// time correlation declaration
double time_corr(int l_max, int frame_count);

// Driver code
int main(int argc, char **argv)
{
    int first_frame;
    cout << "Enter first frame number in integer: " << endl;
    cin >> first_frame;

    int final_frame;
    cout << "Enter final frame number in integer: " << endl;
    cin >> final_frame;

    int digits;
    cout << "Enter total number of digits in filename: " << endl;
    cin >> digits;

    // read image from file
    string directory;
    cout << "Enter data directory in string: " << endl;
    cin >> directory;

    string first_frame_directory = directory + padding(first_frame, digits) + ".bmp";
    string filename = first_frame_directory;

    // open first frame
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    // if fail to read image
    if (img.empty())
    {
        cout << "Error opening image of first frame" << endl;
        return 1;
    }

    // take input of center, inner and outer radii from user
    user_input(img, filename, first_frame_directory);

    // define theta and radii resolution
    size_t m = (int)round(3 * ((params[2] + params[3]) / 2));
    size_t n = 2 * abs(params[2] - params[3]);

    // initialize frame count
    size_t frame_count = 0;

    // create files to store data
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
    myfile.open("gamma.txt");
    myfile << "Each row stores gamma data for each frame." << endl;
    myfile.close();

    // number of modes
    int k_max = 11;
    int l_max = 11;

    for (int i = first_frame; i < final_frame + 1; i++)
    {
        // open image files
        filename = directory + padding(i, digits) + ".bmp";
        Mat img = imread(filename, IMREAD_GRAYSCALE);

        // if fail to read the image
        if (img.empty())
        {
            cout << "Error loading the image " << padding(i, digits) << ".bmp" << endl;
            i++;
        }
        else
        {
            img = preprocessing(&img);

            // declare vectors to store data
            vector<int> contour_X;
            vector<int> contour_Y;
            vector<double> radius_vec;

            // contour detection
            contour_detection(&img, contour_X, contour_Y, radius_vec, m, n);

            // adjustg center for GUV movement
            params[0] = mean(contour_X);
            params[1] = mean(contour_Y);

            // reject stage movement
            if (deviation(radius_vec) > 10)
            { // open image of next frame
                filename = directory + padding(i + 1, digits) + ".bmp";
                Mat img = imread(filename, IMREAD_GRAYSCALE);
                i++;

                if (!img.empty())
                {
                    windowName = "Frame " + padding(i, digits);
                    user_input(img, filename, first_frame_directory);
                }
                else
                { // keep opening next frames until successful
                    while (true)
                    {
                        cout << "Error loading the image " << padding(i, digits) << ".bmp" << endl;
                        filename = directory + padding(i + 1, digits) + ".bmp";
                        Mat img = imread(filename, IMREAD_GRAYSCALE);
                        i++;
                        if (!img.empty())
                        {
                            windowName = "Frame " + padding(i, digits);
                            user_input(img, filename, first_frame_directory);
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
            { // plot contour on the image
                for (int j = 0; j < contour_X.size(); j++)
                {
                    circle(img, Point(contour_X[j], contour_Y[j]), 0, CV_RGB(255, 255, 255), -1);
                }
            }

            // Autocorrelation todo: gpu
            int n = radius_vec.size();
            vector<vector<double>> R(n, vector<double>(n, 0));

            // fill R matrix
            vector<double> radius_vec_copy = radius_vec;
            for (int j = 0; j < n; j++)
            {
                R[0][j] = radius_vec_copy[j];
            }
            for (int j = 0; j < n; j++)
            {
                double t = radius_vec_copy[0];
                radius_vec_copy.erase(radius_vec_copy.begin());
                radius_vec_copy.push_back(t);

                for (int k = 0; k < n; k++)
                {
                    R[j][k] = radius_vec_copy[k];
                }
            }

            // average radius todo: gpu
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
            r_avg /= (3 * (double)n);

            // declare epsilon vector
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

            // store autocorrelation
            myfile.open("epsilon.txt", ios_base::app);
            myfile.setf(ios::scientific);
            myfile.precision(20);

            for (int j = 0; j < n; j++)
            {
                myfile << epsilon[j] << ",";
            }
            myfile << "\n";
            myfile.close();

            vector<double> gamma = linspaceFunc(0, 2 * PI, m);

            // store gamma
            myfile.open("gamma.txt", ios_base::app);
            myfile.setf(ios::scientific);
            myfile.precision(20);

            for (int j = 0; j < n; j++)
            {
                myfile << gamma[j] << ",";
            }
            myfile << "\n";
            myfile.close();

            // Decomposition todo: gpu

            // initialize amplitude arrays
            double a[k_max] = {0};
            double b_prime[l_max] = {0};
            double b[l_max] = {0};

            // amplitude calculation
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
                    b_prime[l] = (b_prime[l - 1] * cos(j * 2 * PI / (double)n) * ((2 * l) - 1) - b_prime[l - 2] * (l - 1)) / l;
                }

                for (int l = 0; l < l_max; l++)
                {
                    b[l] += (2 - (j % 2)) * b_prime[l];
                }
            }
            for (int k = 0; k < k_max; k++)
            {
                a[k] *= 2 / (double)n;
            }
            for (int l = 0; l < l_max; l++)
            {
                b[l] *= PI * ((2 * l) + 1) / (3 * (double)n);
            }

            // store amplitudes
            myfile.open("fourier_amplitudes.txt", ios_base::app);
            myfile.setf(ios::scientific);
            myfile.precision(20);

            for (int k = 0; k < k_max; k++)
            {
                myfile << a[k] << ",";
            }

            myfile << "\n";
            myfile.close();

            myfile.open("legendre_amplitudes.txt", ios_base::app);
            myfile.setf(ios::scientific);
            myfile.precision(20);

            for (int l = 0; l < l_max; l++)
            {
                myfile << b[l] << ",";
            }
            myfile << "\n";
            myfile.close();

            // save image with contour
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
    time_corr(l_max, frame_count);
    return 0;

    // todo: cross correlation
    // todo: spatial correlation
    // todo: bright and dark boundary
    // todo: stage movement option
    // todo: plotting

    // todo: GPU optimization
    // todo: CPU ptimization

}