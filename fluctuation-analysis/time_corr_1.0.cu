#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;

__global__ void hadamard(double *a, double *b, double *c)
{
    int index_1 = threadIdx.x + blockIdx.x * blockDim.x;
    int index_2 = threadIdx.x;

    c[index_1] = a[index_2] * b[index_1];
}

__global__ void mean(double *a, double *b, int frame_count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    a[blockIdx.x] = 0;
    a[blockIdx.x] += b[index] / frame_count;
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
    myfile.close();

    int N = frame_count * frame_count;
    size_t bytes = sizeof(double) * N;
    size_t minibytes = sizeof(double) * frame_count;
    vector<vector<double>> results(l_max, vector<double>(frame_count, 0));

    for (int i = 0; i < l_max; i++)
    {
        vector<double> b_l = vec_data[i];
        vector<double> b_l_copy = vec_data[i];

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
        }

        double *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, minibytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        cudaMemcpy(d_a, b_l.data(), minibytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, time_shift.data(), bytes, cudaMemcpyHostToDevice);

        hadamard<<<frame_count, frame_count>>>(d_a, d_b, d_c);
        cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaMalloc(&d_a, minibytes);
        cudaMalloc(&d_b, bytes);

        cudaMemcpy(d_b, result.data(), bytes, cudaMemcpyHostToDevice);

        mean<<<frame_count, frame_count>>>(d_a, d_b, frame_count);
        cudaMemcpy(results[i].data(), d_a, minibytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);

        myfile.open("time_correlation.txt", ios_base::app);

        for (int j = 0; j < frame_count; j++)
        { // Create an output string stream
            ostringstream streamObj;
            // Set Fixed -Point Notation
            streamObj << fixed;
            // Set precision to 2 digits
            streamObj << setprecision(20);
            // Add double to stream
            streamObj << (results[i][j] * 10000000000000000000);
            cout << results[i][j] * 10000000000000000000 << endl;
            // Get string from output string stream
            string strObj = streamObj.str();
            myfile << strObj + ",";
        }
        myfile << "\n";
        myfile.close();
    }

    return 0;
}