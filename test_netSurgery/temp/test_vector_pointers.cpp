#include <iostream>
#include <vector>

using namespace std;

void modify_func(const vector<int*> &vec_ptrs)
{
    for (int i = 1; i < 10; ++i)
        *vec_ptrs[i] = 10;
}

void print_func(const vector<int*> &vec_ptrs) 
{
    for (int i = 1; i < 10; ++i)
        cout << *vec_ptrs[i] << endl;
}

int main(int argc, const char *argv[])
{
    vector<int*> v_ptrs;
    v_ptrs.reserve(10);
    for (int i=0; i < 10; ++i)
        v_ptrs.push_back(new int(i));

    print_func(v_ptrs);
    modify_func(v_ptrs);
    cout << "after modification: " << endl;
    print_func(v_ptrs);
    
    return 0;
}

