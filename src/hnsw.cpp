#include <hnsw/hnsw.hpp>

size_t HNSW::getRandomLevel(){
    float r = distribution_(rng_); // get a random number in (e-6, 1-e^-6) with uniform distribution
    return (size_t)(-std::log(r)*mult_); 
}


