#ifndef ____Kmeans__
#define ____Kmeans__

#include <iostream>
#include <vector>

using namespace std;

class Kmeans
{
    
private:
    
    vector < vector<float> > training_set;
    
public:
    
    Kmeans(){};
    ~Kmeans(){};
    
    // training function, to get K clustering centers
    void training_function(int K_number, int repeat_number, int cluster_mode);
    
};

#endif /* defined(____Kmeans__) */
