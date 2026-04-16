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
        c[index] = a[index] * b[index];
    }
}

double time_corr(int l_max, int frame_count)
{
    string filename = "legendre_amplitudes.txt";
    ifstream input_file(filename);
    if (!input_file.is_open())
    {
        cerr << "Could not open the file - '"
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
    myfile.open("time_correlation.txt");

    int M = 512; // threads per block
    int N = frame_count * frame_count;
    size_t bytes = sizeof(double) * N;
    vector<vector<double>> results(l_max, vector<double>(frame_count, 0));

    for (int i = 0; i < l_max; i++)
    {
        vector<double> b_l = vec_data[i];
        vector<double> b_l_copy = vec_data[i];

        vector<double> B_l;
        B_l.reserve(N);
        B_l.insert(B_l.end(), b_l.begin(), b_l.end());

        vector<double> time_shift;
        time_shift.reserve(N); // preallocate memory
        time_shift.insert(time_shift.end(), b_l_copy.begin(), b_l_copy.end());

        vector<double> result;
        result.reserve(N);

        for (int j = 1; j < frame_count; j++)
        {
            double t = b_l_copy[0];
            b_l_copy.erase(b_l_copy.begin());
            b_l_copy.push_back(t);
            time_shift.insert(time_shift.end(), b_l_copy.begin(), b_l_copy.end());
            B_l.insert(B_l.end(), b_l.begin(), b_l.end());
        }

        double *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        cudaMemcpy(d_a, B_l.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, time_shift.data(), bytes, cudaMemcpyHostToDevice);

        hadamard<<<(N + M - 1) / M, M>>>(d_a, d_b, d_c, N);
        cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        double sum = 0;
        for (int k = 0; k < N; k++)
        {
            if (k / frame_count == 1)
            {
                results[i][k] = sum / frame_count;
                sum = 0;

                // Create an output string stream
                ostringstream streamObj;
                // Set Fixed -Point Notation
                streamObj << fixed;
                // Set precision to 2 digits
                streamObj << setprecision(20);
                // Add double to stream
                streamObj << results[i][k];
                // Get string from output string stream
                string strObj = streamObj.str();
                myfile << strObj + ",";
            }
            else
            {
                sum += result[k];
            }
        }
        myfile << "\n";
    }
    myfile.close();
    return 0;
}