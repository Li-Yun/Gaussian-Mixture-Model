#include "Kmeans.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <algorithm>
#include<sys/stat.h>
#include <cstdlib>
#include <random>

typedef unsigned char uchar;
using namespace std;

// read training data from the file
vector < vector<float> > read_training_data()
{
    vector < vector <float> > output;
    
    // read data content from the file
    ifstream inputfile("./GMM_dataset.csv", ios::in); // open the file
    string line_1;
    
    while (getline(inputfile, line_1))
    {
        istringstream temp_string(line_1);
        vector <float> col_vector;
        string col_element;
        
        // read every column from the line that is seperated by comma
        while( getline(temp_string, col_element, ',') )
        {
            //float temp_value = stof(col_element);
            float temp_value = atof(col_element.c_str());
            col_vector.push_back(temp_value);
        }
        output.push_back(col_vector);
    }
    
    return output;
}

// initalize the value of clustering seeds randomly
float** setting_clustering_seeds_value(int K_number, int seeds_dimension, vector < vector<float> > training_set)
{
    
    // initialize 2D matrix to store seeds value. Row means the number of seeds, and column means each dimension for each seed
    float** seeds_matrix = new float*[K_number];
    for (int i = 0 ; i < K_number ; i++)
    {
        seeds_matrix[i] = new float[seeds_dimension];
    }
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0, 6);
    
    // randomly assign a value in the range [-6, 6] to each seed
    for (int i = 0 ; i < K_number ; i++){
        for (int j = 0 ; j < seeds_dimension ; j++){
            seeds_matrix[i][j] = dis(gen);
        }
    }
    
    return seeds_matrix;
}

int run_reassign_cluster_center(vector < vector<float> > training_set, int* temp_record_array, int* data_number, int K_number, float** cluster_centers, int empty_cluster_center)
{
    // setting variable
    int new_number_count = 0;
    
    // find the cluster which has the largest sum square error, and assign empty cluster center to that cluster
    float* SSE_array = new float[K_number];
    for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
    {
        if (seed_index == empty_cluster_center)
        {
            SSE_array[seed_index] = 1e-8;
        }
        else
        {
            // compute SSE for each non-empty cluster
            float temp_value = 0.0;
            // go through all data points
            for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
            {
                // compute the distance that the label of data points is equivalent to center's label
                if (temp_record_array[data_index] == seed_index)
                {
                    for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                    {
                        temp_value = temp_value + ( (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) * (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) );
                    }
                }
            }
            SSE_array[seed_index] = temp_value;
        }
    }
    
    // find the cluster center which has max SSE value
    int max_SSE_cluster_center = distance(SSE_array, max_element(SSE_array, SSE_array + K_number));
    
    // randomly generate a new cluster center
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-6.0, 6.0);
    
    float* a_temp_cluster_center = new float[training_set[0].size()];
    for (int i = 0 ; i < training_set[0].size() ; i++)
    {
        a_temp_cluster_center[i] = dis(gen);//(rand() % 17);
    }
    
    // go through all training data
    for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
    {
        // compute the distance between the data point within a certain cluster
        if (temp_record_array[data_index] == max_SSE_cluster_center)
        {
            // compute L2 distance between certain data points and a cluster center with max SSE value
            float L2_value_max_SSE = 0.0;
            for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
            {
                L2_value_max_SSE += ( (training_set[data_index][dimension_index] - cluster_centers[max_SSE_cluster_center][dimension_index]) * (training_set[data_index][dimension_index] - cluster_centers[max_SSE_cluster_center][dimension_index]) );
            }
            L2_value_max_SSE = sqrt(L2_value_max_SSE);
            
            // compute L2 distance between certain data points and a new cluster center
            float L2_value_new_center = 0.0;
            for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
            {
                L2_value_new_center += ( (training_set[data_index][dimension_index] - a_temp_cluster_center[dimension_index]) * (training_set[data_index][dimension_index] - a_temp_cluster_center[dimension_index]) );
            }
            L2_value_new_center = sqrt(L2_value_new_center);
            
            // assign data to the closest cluster center
            if( L2_value_max_SSE < L2_value_new_center )
            {
                temp_record_array[data_index] = max_SSE_cluster_center;
            }
            else if ( L2_value_new_center < L2_value_max_SSE )
            {
                temp_record_array[data_index] = empty_cluster_center;
            }
            else
            {
                // generate a random number
                float random_number = ((float) rand() / (RAND_MAX)) + 1;
                
                if(random_number >= 0.5)
                {
                    temp_record_array[data_index] = max_SSE_cluster_center;
                }
                else
                {
                    temp_record_array[data_index] = empty_cluster_center;
                    
                }
            }
        }
    }
    
    // go through all training data points
    for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
    {
        if (temp_record_array[data_index] == empty_cluster_center)
        {
            new_number_count++;
        }
    }
    
    return new_number_count;
}

// training function, and find K clustering centers given training set
void Kmeans::training_function(int K_number, int repeat_number, int cluster_mode)
{
    // read training data from the file
    training_set = read_training_data();
    
    // declare 3D matrix to store a set of seed's matrix
    float*** cluster_centers_set= new float **[K_number];
    for(int i = 0; i < K_number; i++)
    {
        cluster_centers_set[i] = new float* [training_set[0].size()];
        for(int j = 0; j < training_set[0].size() ; j++)
        {
            cluster_centers_set[i][j] = new float[repeat_number];
        }
    }
    
    // declare a temp 2D matrix to store final record array
    float** final_record_array = new float*[repeat_number];
    for (int i = 0 ; i < repeat_number ; i++)
    {
        final_record_array[i] = new float[training_set.size()];
    }
    
    // repeat clustering algorithm multiple times
    for (int repeat_index = 0 ; repeat_index < repeat_number ; repeat_index++)
    {
        cout << "Repeat Number: " << repeat_index << endl;
        
        // declare a temp record array to store which each data point belongs to its the closest cluster
        int* temp_record_array = new int[training_set.size()];
        
        // declare a temp 2D matrix to store the sum of data points for each cluster
        float** new_cluster_matrix = new float*[K_number];
        for (int i = 0 ; i < K_number ; i++)
        {
            new_cluster_matrix[i] = new float[training_set[0].size()];
        }
        
        // Step 1: randomly initialize K clustering centers
        float** cluster_centers = setting_clustering_seeds_value(K_number, training_set[0].size(), training_set);
        
        
        // store all cluster centers to the file
        string output_file_name = "./all_cluster_center" + to_string(repeat_index) + ".csv";
        ofstream seeds(output_file_name);
        
        // clustering algorithm: Kmeans
        for (int iteration_index = 1 ; iteration_index <= 2000 ; iteration_index++)
        {
            // Step 2-1: for each cluster, it compoute the distance between each cluster's position and assign each data point to its the closest cluster.
            for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
            {
                float* distance_between_data_seed = new float[K_number];
                
                for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
                {
                    // compute L2 distance between each data point and each cluster center
                    float L2_value = 0.0;
                    for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                    {
                        L2_value += ( (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) * (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) );
                    }
                    L2_value = sqrt(L2_value);
                    // store each distance value into distance vector
                    //distance_between_data_seed.push_back(L2_value);
                    distance_between_data_seed[seed_index] = L2_value;
                    //cout << "===================" << endl;
                    //cout << distance_between_data_seed[seed_index] << " ";
                }
                
                // store distance value into vector, and prepare to find the closest cluster.
                vector <float> distance_vector(distance_between_data_seed, distance_between_data_seed + K_number);
                
                // find the corresponding cluster center by applying distance information
                float min_value = *std::min_element(distance_vector.begin(), distance_vector.end());
                int min_pos = distance(distance_vector.begin(), min_element(distance_vector.begin(), distance_vector.end()));
                
                // store the corresponding seed to seed record vector
                temp_record_array[data_index] = min_pos;
                
                // free memory
                delete[] distance_between_data_seed;
            }
            
            // Step 2-2: compute the number of data points in each cluster
            int* data_number = new int[K_number];
            for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
            {
                int number_count = 0;
                // go through all training data points
                for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
                {
                    if (temp_record_array[data_index] == seed_index)
                    {
                        number_count++;
                    }
                }
                // if the number of data points in one cluster is zero, we need to re-assign the cluster center
                if(number_count == 0)
                {
                    // choose different mode for empty cluster
                    if ( cluster_mode == 0 )
                    {
                        // re-assign cluster center function
                        number_count = run_reassign_cluster_center(training_set, temp_record_array, data_number, K_number, cluster_centers, seed_index);
                        
                        if (number_count == 0)
                        {
                            number_count = 1;
                        }
                    }
                    else if ( cluster_mode == 1 )
                    {
                        random_device rd;
                        mt19937 gen(rd());
                        uniform_real_distribution<float> dis(-6.0, 6.0);
                        
                        // randomly re-generate a new cluster center
                        for (int i = 0 ; i < training_set[0].size() ; i++)
                        {
                            cluster_centers[seed_index][i] = dis(gen);
                        }
                        number_count = 1;
                    }
                }
                data_number[seed_index] = number_count;
            }
            
            // Step 2-3: update cluster center using seed record vector
            for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
            {
                // go through all training data points
                for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
                {
                    if (temp_record_array[data_index] == seed_index)
                    {
                        // update each dimension values for each cluster
                        for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                        {
                            new_cluster_matrix[seed_index][dimension_index] = new_cluster_matrix[seed_index][dimension_index] + training_set[data_index][dimension_index];
                        }
                    }
                }
                
                // compute new cluster centers
                for (int data_dimention = 0 ; data_dimention < training_set[0].size() ; data_dimention++)
                {
                    new_cluster_matrix[seed_index][data_dimention] = new_cluster_matrix[seed_index][data_dimention] / float(data_number[seed_index]);
                }
            }
            
            // Pre-computing for stop condition 1: compute the distance between previous centers and new centers
            int seed_count = 0;
            for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
            {
                int true_number = 0;
                
                for (int data_dimention = 0 ; data_dimention < training_set[0].size() ; data_dimention++)
                {
                    if (fabs(new_cluster_matrix[seed_index][data_dimention] - cluster_centers[seed_index][data_dimention]) <= 1e-15 )
                    {
                        true_number++;
                    }
                }
                
                if (true_number == training_set[0].size() - 1)
                {
                    seed_count++;
                }
            }
            
            // Pre-computing for stop condition 2: compute previous SSE value
            float previous_SSE_value = 0.0;
            for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
            {
                float temp_value = 0.0;
                // go through all data points
                for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
                {
                    // compute the distance that the label of data points is equivalent to center's label
                    if (temp_record_array[data_index] == seed_index)
                    {
                        for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                        {
                            temp_value = temp_value + ( (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) * (training_set[data_index][dimension_index] - cluster_centers[seed_index][dimension_index]) );
                        }
                    }
                }
                previous_SSE_value = previous_SSE_value + temp_value;
            }
            
            // Pre-computing for stop condition 3: compute sum squared error (SSE) value
            float SSE_value = 0.0;
            for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
            {
                float temp_value = 0.0;
                // go through all data points
                for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
                {
                    // compute the distance that the label of data points is equivalent to center's label
                    if (temp_record_array[data_index] == seed_index)
                    {
                        for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                        {
                            temp_value = temp_value + ( (training_set[data_index][dimension_index] - new_cluster_matrix[seed_index][dimension_index]) * (training_set[data_index][dimension_index] - new_cluster_matrix[seed_index][dimension_index]) );
                        }
                    }
                }
                SSE_value = SSE_value + temp_value;
            }
            
            // stop conditions
            if( seed_count == K_number || fabs(SSE_value - previous_SSE_value) <= 1e-10 )
            {
                cout << "Total Iteartion: " << iteration_index << endl;
                cout << "Training Stage is done!!" << endl;
                
                // store temp record value to final record matrix
                for (int j = 0 ; j < training_set.size() ; j++)
                {
                    final_record_array[repeat_index][j] = temp_record_array[j];
                }
                
                // store clusters
                for (int i = 0 ; i < K_number ; i++)
                {
                    for (int j = 0 ; j < training_set[0].size() ; j++)
                    {
                        if (j == training_set[0].size() - 1){
                            seeds << new_cluster_matrix[i][j];
                        }
                        else{
                            seeds << new_cluster_matrix[i][j]<< ",";
                        }
                    }
                    seeds << "\n";
                }
                seeds.close();
                
                // store each center matrix to 3D vector
                for(int i = 0 ; i < K_number ; i++)
                {
                    for(int j = 0 ; j < training_set[0].size() ; j++)
                    {
                        cluster_centers_set[i][j][repeat_index] = new_cluster_matrix[i][j];
                    }
                }
                break;
            }
            
            // reassign new cluster matrix to as cluster center (previous cluster matrix)
            for (int i = 0 ; i < K_number ; i++)
            {
                for (int j = 0 ; j < training_set[0].size() ; j++)
                {
                    cluster_centers[i][j] = new_cluster_matrix[i][j];
                }
            }
            
            // store clusters
            for (int i = 0 ; i < K_number ; i++)
            {
                for (int j = 0 ; j < training_set[0].size() ; j++)
                {
                    if (j == training_set[0].size() - 1){
                        seeds << cluster_centers[i][j];
                    }
                    else{
                        seeds << cluster_centers[i][j]<< ",";
                    }
                }
                seeds << "\n";
            }
            
        }
    }
    
    // compute SSE value of each cluster center
    vector <float> SSE_vector;
    for (int repeat_index = 0 ; repeat_index < repeat_number ; repeat_index++)
    {
        float temp_SSE_value = 0.0;
        for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
        {
            for (int data_index = 0 ; data_index < training_set.size() ; data_index++)
            {
                // compute the distance that the label of data points is equivalent to center's label
                if (final_record_array[repeat_index][data_index] == seed_index)
                {
                    for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
                    {
                        temp_SSE_value += ( (training_set[data_index][dimension_index] - cluster_centers_set[seed_index][dimension_index][repeat_index] ) * ( training_set[data_index][dimension_index] - cluster_centers_set[seed_index][dimension_index][repeat_index] ) );
                    }
                }
            }
        }
        SSE_vector.push_back(temp_SSE_value);
    }
    
    // choose minimum SSE value from SSE vector
    float min_SSE = *std::min_element(SSE_vector.begin(), SSE_vector.end());
    int min_SSE_pos = distance(SSE_vector.begin(), min_element(SSE_vector.begin(),SSE_vector.end()));
    
    // compute sum squared separation given the best cluster centers
    float SSS_value = 0.0;
    for (int seed_index = 0 ; seed_index < K_number ; seed_index++)
    {
        for (int seed_index_2 = seed_index + 1 ; seed_index_2 < K_number ; seed_index_2++)
        {
            for (int dimension_index = 0 ; dimension_index < training_set[0].size() ; dimension_index++)
            {
                SSS_value = SSS_value + ( (cluster_centers_set[seed_index][dimension_index][min_SSE_pos] - cluster_centers_set[seed_index_2][dimension_index][min_SSE_pos]) * (cluster_centers_set[seed_index][dimension_index][min_SSE_pos] - cluster_centers_set[seed_index_2][dimension_index][min_SSE_pos]) );
            }
        }
    }
    
    // display some results
    cout << "=========================================" << endl;
    cout << "Sum Squared Error value: " << min_SSE << endl;
    cout << "Sum Squared Separation value: " << SSS_value << endl;
    cout << "=========================================" << endl;
    
    // store the minimum SSE position
    string output_file_name = "./min_sse_pos.csv";
    ofstream seeds(output_file_name);
    seeds << min_SSE_pos;
    seeds.close();
    
}

