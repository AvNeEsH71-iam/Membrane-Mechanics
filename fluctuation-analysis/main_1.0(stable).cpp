// Interlacing and horizontal scan

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

const double PI = 3.14159265358979323846;

int params[4] = {0}; /*   0 - x coordinate of center
                          1 - y coordinate of center
                          2 - inner radius
                          3 - outer radius   */

string windowName = "Frame";

// mean
template <typename T>
T mean(vector<T> data)
{
    T sum = 0;
    for (int i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    return (T)(sum / data.size());
}

// deviation
double deviation(vector<double> radius_vec)
{
    size_t N = radius_vec.size();
    double sum = 0;
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

// contour detection todo: gpu
void contour_detection(Mat *img, vector<int> &contour_X, vector<int> &contour_Y, vector<double> &radius_vec, int mode)
{
    Mat arr_data = *img;

    int x_c = params[0];
    int y_c = params[1];

    int y_min, y_max, x_min, x_max;
    int min = 255;

    // bounding box
    for (int i = y_c - params[3]; i < y_c - params[2]; i++)
    {
        int intensity = arr_data.at<uchar>(i, x_c);
        if (intensity < min)
        {
            min = intensity;
            y_min = i;
        }
    }

    min = 255;
    for (int i = y_c + params[2]; i < y_c + params[3]; i++)
    {
        int intensity = arr_data.at<uchar>(i, x_c);
        if (intensity < min)
        {
            min = intensity;
            y_max = i;
        }
    }

    min = 255;
    for (int i = x_c - params[3]; i < x_c - params[2]; i++)
    {
        int intensity = arr_data.at<uchar>(y_c, i);
        if (intensity < min)
        {
            min = intensity;
            x_min = i;
        }
    }

    min = 255;
    for (int i = x_c + params[2]; i < x_c + params[3]; i++)
    {
        int intensity = arr_data.at<uchar>(y_c, i);
        if (intensity < min)
        {
            min = intensity;
            x_max = i;
        }
    }

    // peak detection top
    for (int i = x_c - params[2] + 1; i < x_c + params[2]; i++)
    {
        min = 255;
        int index;
        for (int j = y_min + 1; j < y_c - params[2]; j++)
        {
            if ((mode == 0) && (j % 2 == 0))
            {
                int intensity = arr_data.at<uchar>(j, i);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }
            if ((mode == 1) && (j % 2 == 1))
            {
                int intensity = arr_data.at<uchar>(j, i);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }
        }

        contour_X.push_back(i);
        contour_Y.push_back(index);
        radius_vec.push_back(sqrt(pow(index - y_c, 2) + pow(i - x_c, 2)));
    }
    // peak detection middle left
    for (int i = y_c - params[2] + 1; i < y_c + params[2]; i++)
    {
        if ((mode == 0) && (i % 2 == 0))
        {
            min = 255;
            int index;
            for (int j = x_min + 1; j < x_c - params[2]; j++)
            {
                int intensity = arr_data.at<uchar>(i, j);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }

            contour_X.push_back(index);
            contour_Y.push_back(i);
            radius_vec.push_back(sqrt(pow(i - y_c, 2) + pow(index - x_c, 2)));
        }
        if ((mode == 1) && (i % 2 == 1))
        {
            min = 255;
            int index;
            for (int j = x_min + 1; j < x_c - params[2]; j++)
            {
                int intensity = arr_data.at<uchar>(i, j);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }

            contour_X.push_back(index);
            contour_Y.push_back(i);
            radius_vec.push_back(sqrt(pow(i - y_c, 2) + pow(index - x_c, 2)));
        }
    }
    // peak detection middle right
    for (int i = y_c - params[2] + 1; i < y_c + params[2]; i++)
    {
        if ((mode == 0) && (i % 2 == 0))
        {
            min = 255;
            int index;
            for (int j = x_c + params[2] + 1; j < x_max; j++)
            {
                int intensity = arr_data.at<uchar>(i, j);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }

            contour_X.push_back(index);
            contour_Y.push_back(i);
            radius_vec.push_back(sqrt(pow(i - y_c, 2) + pow(index - x_c, 2)));
        }
        if ((mode == 1) && (i % 2 == 1))
        {
            min = 255;
            int index;
            for (int j = x_c + params[2] + 1; j < x_max; j++)
            {
                int intensity = arr_data.at<uchar>(i, j);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }

            contour_X.push_back(index);
            contour_Y.push_back(i);
            radius_vec.push_back(sqrt(pow(i - y_c, 2) + pow(index - x_c, 2)));
        }
    }
    // peak detection bottom
    for (int i = x_c - params[2] + 1; i < x_c + params[2]; i++)
    {
        min = 255;
        int index;
        for (int j = y_c + params[2] + 1; j < y_max; j++)
        {
            if ((mode == 0) && (j % 2 == 0))
            {
                int intensity = arr_data.at<uchar>(j, i);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }
            if ((mode == 1) && (j % 2 == 1))
            {
                int intensity = arr_data.at<uchar>(j, i);

                if (intensity < min)
                {
                    min = intensity;
                    index = j;
                }
            }
        }

        contour_X.push_back(i);
        contour_Y.push_back(index);
        radius_vec.push_back(sqrt(pow(index - y_c, 2) + pow(i - x_c, 2)));
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
void user_input(Mat img, string filename, string first_frame_directory)
{
    // Create a window
    namedWindow(windowName, 1);

    // set the callback function for any mouse event
    setMouseCallback(windowName, click_event, &filename);

    // show the image
    imshow(windowName, img);

    if (filename == first_frame_directory)
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

    // Read image from file
    string directory;
    cout << "Enter data directory in string: " << endl;
    cin >> directory;

    string first_frame_directory = directory + to_string(first_frame) + ".bmp";
    string filename = first_frame_directory;
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    user_input(img, filename, first_frame_directory);

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
    myfile.open("gamma.txt");
    myfile << "Each row stores gamma data for each frame." << endl;
    myfile.close();

    // number of modes
    int k_max = 11;
    int l_max = 11;

    for (int i = first_frame; i < final_frame + 1; i++)
    {
        filename = directory + to_string(i) + ".bmp";
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
                vector<int> contour_X;
                vector<int> contour_Y;
                vector<double> radius_vec;
                vector<double> theta_vec;

                contour_detection(&img, contour_X, contour_Y, radius_vec, mode);
                params[0] = mean(contour_X);
                params[1] = mean(contour_Y);

                // rejecting stage movement
                if (deviation(radius_vec) > 7)
                {
                    filename = directory + to_string(i + 1) + ".bmp";
                    Mat img = imread(filename, IMREAD_GRAYSCALE);
                    i++;

                    if (!img.empty())
                    {
                        windowName = "Frame" + to_string(i);
                        user_input(img, filename, first_frame_directory);
                    }
                    else
                    {
                        while (true)
                        {
                            cout << "Error loading the image " << to_string(i) << ".bmp" << endl;
                            filename = directory + to_string(i + 1) + ".bmp";
                            Mat img = imread(filename, IMREAD_GRAYSCALE);
                            i++;
                            if (!img.empty())
                            {
                                windowName = "Frame" + to_string(i);
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
                {
                    for (int j = 0; j < contour_X.size(); j++)
                    {
                        circle(img, Point(contour_X[j], contour_Y[j]), 0, CV_RGB(255, 255, 255), -1);
                    }
                }

                // angular sorting todo: gpu
                for (int j = 0; j < contour_X.size(); j++)
                {
                    int x = contour_X[j] - params[0];
                    int y = -(contour_Y[j] - params[1]);
                    if ((x > 0) && (y > 0))
                    {
                        theta_vec.push_back(atan(y / (double)x));
                    }
                    else if ((x > 0) && (y < 0))
                    {
                        theta_vec.push_back(atan(y / (double)x) + 2 * PI);
                    }
                    else if ((x < 0) && (y > 0))
                    {
                        theta_vec.push_back(atan(y / (double)x) + PI);
                    }
                    else
                    {
                        theta_vec.push_back(atan(y / (double)x) + PI);
                    }
                }

                for (int j = 0; j < theta_vec.size(); j++)
                {
                    for (int k = j + 1; k < theta_vec.size(); k++)
                    {
                        if (theta_vec[k] < theta_vec[j])
                        {
                            double temp_1 = theta_vec[j];
                            theta_vec[j] = theta_vec[k];
                            theta_vec[k] = temp_1;

                            temp_1 = radius_vec[j];
                            radius_vec[j] = radius_vec[k];
                            radius_vec[k] = temp_1;

                            int temp_2 = contour_X[j];
                            contour_X[j] = contour_X[k];
                            contour_X[k] = temp_2;

                            temp_2 = contour_X[j];
                            contour_Y[j] = contour_Y[k];
                            contour_Y[k] = temp_2;
                        }
                    }
                }

                // Autocorrelation todo: gpu
                int n = radius_vec.size();
                double R[n][n] = {0};

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

                vector<double> gamma;
                for (int j = 0; j < n; j++)
                {
                    gamma.push_back(theta_vec[j] - theta_vec[0]);
                }
                // storing gamma
                myfile.open("gamma.txt", ios_base::app);

                for (int j = 0; j < n; j++)
                {
                    myfile << to_string(gamma[j]) + ",";
                }
                myfile << "\n";
                myfile.close();

                // Decomposition todo: gpu

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
                        b_prime[l] = (b_prime[l - 1] * cos(j * 2 * PI / n) * (2 * l - 1) - b_prime[l - 2] * (l - 1)) / l;
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
    time_corr(l_max, 2 * frame_count);
    return 0;

    // todo: corner points
    // todo: cross correlation
    // todo: spatial correlation

    // todo: GPU optimization
    // todo: split code
}