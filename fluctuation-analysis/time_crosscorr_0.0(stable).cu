#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;

__global__ void hadamard(double *a, double *b, double *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        c[index] = 0;
        for (int k = 0; k < n; k++)
        {
            c[index] += (a[k] * b[index + k]) / (double)n;
        }
    }
}

double time_crosscorr(int l_max, int frame_count)
{
    // legendre amplitude input

    string filename = "legendre_amplitudes.txt";
    ifstream input_file_1(filename);
    if (!input_file_1.is_open())
    {
        cout << "Could not open the file - '"
             << filename << "'" << endl;
        return EXIT_FAILURE;
    }

    vector<vector<double>> vec_data(l_max, vector<double>(frame_count, 0));
    int i = 0;
    string line;
    bool start = true;
    while (getline(input_file_1, line))
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

    input_file_1.close();

    // time_cross_correlation, input of required combination of modes

    filename = "time_cross_input.txt";
    ifstream input_file_2(filename);
    if (!input_file_2.is_open())
    {
        cout << "Could not open the file - '"
             << filename << "'" << endl;
        return EXIT_FAILURE;
    }

    vector<vector<int>> time_cross_input;
    i = 0;
    while (getline(input_file_2, line))
    {
        string token;
        stringstream ss(line);
        int j = 0;
        vector<int> vec_comb;
        while (getline(ss, token, ','))
        {
            if (stoi(token))
            {
                vec_comb.push_back(stoi(token));
                j++;
            }
        }
        i++;
        time_cross_input.push_back(vec_comb);
    }

    input_file_2.close();

    ofstream myfile;
    myfile.open("time_cross_correlation.txt");
    myfile.setf(ios::scientific);
    myfile.precision(20);

    // time cross correlation calculation

    int M = 512; // threads per block
    size_t bytes = sizeof(double) * frame_count;
    size_t minibytes = sizeof(double) * (frame_count / (int)2);

    for (int i = 0; i < time_cross_input.size(); i++)
    {
        vector<double> b_1 = vec_data[time_cross_input[i][0]];
        vector<double> b_2 = vec_data[time_cross_input[i][1]];

        vector<double> result;
        result.reserve(frame_count / (int)2);

        double *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, minibytes);

        cudaMemcpy(d_a, b_1.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_2.data(), bytes, cudaMemcpyHostToDevice);

        hadamard<<<(frame_count + M - 1) / M, M>>>(d_a, d_b, d_c, frame_count / (int)2);
        cudaMemcpy(result.data(), d_c, minibytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        for (int j = 0; j < frame_count / (int)2; j++)
        {
            myfile << result[j] << ",";
        }

        myfile << "\n";
    }

    myfile.close();
    return 0;
}