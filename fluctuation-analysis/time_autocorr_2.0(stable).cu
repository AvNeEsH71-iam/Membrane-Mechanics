#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;

__global__ void hadamard(double *a, double *b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        b[index] = 0;
        for (int k = 0; k < n; k++)
        {
            b[index] += (a[k] * a[index + k]) / (double)n;
        }
    }
}

double time_autocorr(int l_max, int frame_count)
{

    string filename = "legendre_amplitudes.txt";
    ifstream input_file(filename);
    if (!input_file.is_open())
    {
        cout << "Could not open the file - '"
             << filename << "'" << endl;
        return EXIT_FAILURE;
    }

    vector<vector<double>> vec_data(l_max, vector<double>(frame_count, 0));
    int i = 0;
    string line;
    bool start = true;
    while (getline(input_file, line))
    {
        if (!start)
        {
            string token;
            stringstream ss(line);
            int j = 0;
            while (getline(ss, token, ','))
            {
                if (stod(token))
                {
                    vec_data[j][i] = stod(token);
                    j++;
                }
            }
            i++;
        }
        else
        {
            start = false;
        }
    }

    input_file.close();

    ofstream myfile;
    myfile.open("time_auto_correlation.txt");
    myfile.setf(ios::scientific);
    myfile.precision(20);

    int M = 512; // threads per block
    size_t bytes = sizeof(double) * frame_count;
    size_t minibytes = sizeof(double) * (frame_count / (int)2);

    for (int i = 0; i < l_max; i++)
    {
        vector<double> b_l = vec_data[i];

        vector<double> result;
        result.reserve(frame_count / (int)2);

        double *d_a, *d_b;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, minibytes);

        cudaMemcpy(d_a, b_l.data(), bytes, cudaMemcpyHostToDevice);

        hadamard<<<(frame_count + M - 1) / M, M>>>(d_a, d_b, frame_count / (int)2);
        cudaMemcpy(result.data(), d_b, minibytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);

        for (int j = 0; j < frame_count / (int)2; j++)
        {
            myfile << result[j] << ",";
        }

        myfile << "\n";
    }

    myfile.close();
    return 0;
}