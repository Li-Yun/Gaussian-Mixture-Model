#include <iostream>
#include "Kmeans.h"
using namespace std;

int main(int argc, char* argv[])
{
    // setting variables
    int seeds_number = 0;
    int repeat_number = 0;
    int cluster_mode = 0;
    int seed_mode = 0;
    
    // check argument format
    if (argc <= 1)
    {
        cout << "Error Format." << endl;
        cout << "Argument Format in this program is: (1)./a.out training_kmeans K_number Repeat_number or (2)./a.out classify_data" << endl;
    }
    else
    {
        // get input array, and convert to string
        string name_string(argv[1]);
        
        // string checking
        if (name_string.compare("training_kmeans") == 0){
            cout << "Welcome to Kmeans Program:" << endl;
            cout << "Argument 1: ./a.out" << endl;
            cout << "Argument 2: training_kmeans or classify_data (does not have other parameters)" << endl;
            cout << "Argument 3: K Number." << endl;
            cout << "Argument 4: Repeat Number." << endl;
            cout << "Argument 5: Mode for dealing with empty cluster. 0 is the method of find maximum SSE. 1 is randomly re-assign a new centroid." << endl;
            
            // get K seeds
            seeds_number = atoi(argv[2]);
            
            // get repeat number
            repeat_number = atoi(argv[3]);
            
            // different mode for dealing with empty cluster
            cluster_mode = atoi(argv[4]);
            
            // create Kmeans object, and do training stage
            Kmeans* kmean = new Kmeans();
            kmean -> training_function(seeds_number, repeat_number, cluster_mode);
            
        }
        else{
            cout << "Error Input!!" << endl;
        }
    }
    return 0;
}
