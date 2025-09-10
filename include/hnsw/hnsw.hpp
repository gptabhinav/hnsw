#ifndef HNSW_HPP
#define HNSW_HPP

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>

class HNSW{
    public:
        HNSW(size_t M, size_t efConstruction, size_t dimension, bool use_heuristic = true):
            M_(M),
            maxM_(M),
            maxM0_(2*M),
            efConstruction_(efConstruction),
            dimension_(dimension),
            mult_(1.0/std::log(M_)),
            use_heuristic_(use_heuristic),
            maxlevel_(0),
            enterpoint_(0)
            {
                // Constructor implementation
            }   

        // Core API methods
        void addPoint(const std::vector<float>& point, int label);
        std::vector<std::pair<float, int>> searchKNN(const std::vector<float>& query, size_t k, size_t ef) const;

        // Index I/O methods
        void saveIndex(const std::string &filename) const;
        void loadIndex(const std::string &filename);

        // Utility methods
        size_t getCurrentCount() const { return points_.size(); }
        size_t getMaxLevel() const { return maxlevel_; }
        bool isIndexEmpty() const { return points_.empty(); }


    private:
    // private member variables are usually named with a trailing underscore
        
        // Core parameters
        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t efConstruction_;

        // *** Affects algorithm behavior ***
        // Added for educational purposes
        // *** This is not a standard parameter in HNSW implementations ***
        // It controls whether to use heuristic for neighbor selection
        // In practice, this is usually set to true
        // If set to false, the algorithm will use a simpler method for neighbor selection
        // This can affect the quality of the graph and search performance
        // In general, using the heuristic leads to better performance
        bool use_heuristic_;

        // Level generation parameters
        std::mt19937 rng_; // this is just a random number generator
        std::uniform_real_distribution<float> distribution_{1e-6, 1.0-1e-6}; // uniform distribution between e^-6 and 1-e^-6
        // because log(0) is undefined and we want to avoid it, also avoid 1 to prevent log(1)=0
        // the reason we avoid 1 is that log(1) = 0, and assigns the lowest level possible and affects the distribution
        double mult_; // normalization factor for level generation, defaults to 1/ln(M)

        // Graph structure
        size_t dimension_; // dimensionality of the data points
        std::vector<std::vector<float>> points_;   // data points
        std::vector<std::vector<std::vector<size_t>>> layers_; // 3D vector: level -> node -> neighbors
        std::vector<size_t> element_levels_; // level of each element
        std::vector<int> labels_; // mapping from index to label
        std::unordered_map<int, size_t> label_lookup_; // mapping from label to index


        size_t maxlevel_;
        size_t enterpoint_;

        // Private helper methods
        size_t getRandomLevel();
        float computeDistance(const std::vector<float>& a, const std::vector<float>& b) const;
        std::vector<size_t> searchLayer(const std::vector<float>& query, const std::vector<size_t>& entry_points, size_t ef, size_t level) const;
        std::vector<size_t> selectNeighbors(size_t node_id, const std::vector<std::pair<float, size_t>>& candidates, size_t M) const;

};

#endif // HNSW_HPP