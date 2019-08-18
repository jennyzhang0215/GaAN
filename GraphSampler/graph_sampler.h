#ifndef GRAPH_SAMPLER_H_
#define GRAPH_SAMPLER_H_
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <utility>

#define ASSERT(x) if(!(x)) {std::cout << "Line:" << __LINE__ << " " #x << " does not hold!" << std::endl;exit(0);}

namespace graph_sampler {
typedef std::unordered_map<int, std::unordered_set<int> > GRAPH_DATA_T;
typedef std::mt19937 RANDOM_ENGINE;
const static int MAX_RANDOM_ENGINE_NUM = 128;
const static int MAX_ALLOWED_NODE = std::numeric_limits<int>::max();
const static long long MAX_ALLOWED_EDGE = std::numeric_limits<long long>::max();

class SimpleGraph {
public:
  SimpleGraph() {}
	SimpleGraph(bool undirected,
		        int max_node_num = std::numeric_limits<int>::max(),
		        long long max_edge_num = std::numeric_limits<long long>::max()) {
        set_undirected(undirected);
        set_max(max_node_num, max_edge_num);
	}

    void set_undirected(bool undirected) {
        undirected_ = undirected;
    }

    void set_max(int max_node_num,
                 long long max_edge_num) {
        max_node_num_ = max_node_num;
        max_edge_num_ = max_edge_num;
        if (max_node_num_ < 0) max_node_num_ = MAX_ALLOWED_NODE;
        if (max_edge_num_ < 0) max_edge_num_ = MAX_ALLOWED_EDGE;
    }

    bool undirected() const { return undirected_; }
    int node_num() const { return node_num_; }
    int edge_num() const { return edge_num_; }
    const GRAPH_DATA_T* data() const { return &data_; }

	bool is_full() {
		return edge_num_ >= max_edge_num_ || node_num_ >= max_node_num_;
	}

    bool has_node(int node) {
        return data_.find(node) != data_.end();
    }

    bool insert_new_node(int node) {
		if (is_full()) {
			return false;
		}
		GRAPH_DATA_T::iterator node_it = data_.find(node);
		if (node_it == data_.end()) {
			data_[node] = std::unordered_set<int>();
			node_num_++;
		}
		return true;
    }

    bool insert_new_edge(std::pair<int, int> edge) {
        if(is_full()) {
            return false;
        }
		// 1. Insert the start point and end point
		GRAPH_DATA_T::iterator start_node_it = data_.find(edge.first);
		GRAPH_DATA_T::iterator end_node_it = data_.find(edge.second);
		bool has_insert_start = false;
		if (start_node_it == data_.end()) {
			has_insert_start = true;
			data_[edge.first] = std::unordered_set<int>();
            node_num_++;
			start_node_it = data_.find(edge.first);
		}
		if(end_node_it == data_.end()) {
			// Deal with the special case that the graph will be full after inserting the first node
			if (has_insert_start && is_full()) {
				data_.erase(start_node_it);
                node_num_--;
				return false;
			}
			data_[edge.second] = std::unordered_set<int>();
            node_num_++;
			end_node_it = data_.find(edge.second);
		}
        if (edge.second == edge.first) return true; // Return if the edge is a self-loop
		if(start_node_it->second.find(edge.second) == start_node_it->second.end()) {
			start_node_it->second.insert(edge.second);
			edge_num_++;
		}
		if (undirected_) {
			if (end_node_it->second.find(edge.first) == end_node_it->second.end()) {
				end_node_it->second.insert(edge.first);
			}
		}
        return true;
    }

    bool insert_nodes(const std::vector<int> &ids) {
		std::vector<int> inserted_ids;
		for (int id: ids) {
			if(is_full()) {
				for(int insert_id: inserted_ids) {
					data_.erase(insert_id);
				}
				return false;
			}
            if(data_.find(id) != data_.end()) {
                continue;
            } else {
				inserted_ids.push_back(id);
                data_[id] = std::unordered_set<int>();
                node_num_++;
            }
        }
        return true;
    }

    void convert_to_csr(std::vector<int> *end_points,
		                std::vector<int> *ind_ptr,
		                std::vector<int> *node_ids,
		                const int* src_node_ids,
		                int src_node_size) {
        int shift = 0;
        std::unordered_map<int, int> node_id_map;
        int counter = 0;
        for(const auto& ele: data_) {
            node_id_map[ele.first] = counter;
            counter++;
        }
        for (const auto &ele: data_) {
            node_ids->push_back(src_node_ids[ele.first]);
            ind_ptr->push_back(shift);
            for (int node: ele.second) {
                end_points->push_back(node_id_map[node]);
				shift++;
            }
        }
        ind_ptr->push_back(shift);
    }
private:
    int max_node_num_ = MAX_ALLOWED_NODE;
    long long max_edge_num_ = MAX_ALLOWED_EDGE;
    int node_num_ = 0;
    long long edge_num_ = 0;
    GRAPH_DATA_T data_;
    bool undirected_ = true;
};

class GraphSampler {
public:
GraphSampler(int seed_id=-1) {
	set_seed(seed_id);
}

void set_seed(int seed_id) {
	std::vector<std::uint32_t> seeds(MAX_RANDOM_ENGINE_NUM);
  int u_seed_id = seed_id;
	if(seed_id < 0) {
		//Randomly set seed of the engine
		std::random_device rd;
		std::uniform_int_distribution<int> dist(0, 100000);
		u_seed_id = dist(rd);	
	}
  RANDOM_ENGINE base_engine;
  base_engine.seed(u_seed_id);
  std::unordered_map<int, int> pool;
  for(int i = 0; i < MAX_RANDOM_ENGINE_NUM; i++) {
    std::uniform_int_distribution<int> dist(i, 100000000);
    int val = dist(base_engine);
    if(pool.find(val) != pool.end()) {
      eng_[i].seed(pool[val]);
    } else {
      eng_[i].seed(val);
    }
    if(pool.find(i) != pool.end()) {
      pool[val] = pool[i];
    } else {
      pool[val] = i;
    }
  }
}

/*
Sampling the graph by randomwalk.
At every step, we will return to the original node with return_p. Otherwise, we will jump randomly to a conneted node.
See [KDD06] Sampling from Large Graphs
------------------------------------------------------------------
Params:
src_end_points: end points in the source graph
src_ind_ptr: ind ptr in the source graph
src_node_ids: node ids of the source graph
src_undirected: whether the source graph is undirected
src_node_num: number of nodes in the source graph
initial_node: initial node of the random walk, if set to negative, the initial node will be chosen randomly from the original graph
walk_length: length of the random walk
return_prob: the returning probability
max_node_num: the maximum node num allowed in the sampled subgraph
max_edge_num: the maximum edge num allowed in the sampled subgraph
------------------------------------------------------------------
Return:
subgraph: the sampled graph
*/
SimpleGraph* random_walk(const int* src_end_points,
	                       const int* src_ind_ptr,
	                       const int* src_node_ids,
                         bool src_undirected,
	                       int src_node_num,
	                       int initial_node,
	                       int walk_length=10,
	                       double return_prob=0.15,
                         int max_node_num=std::numeric_limits<int>::max(),
	                       long long max_edge_num = std::numeric_limits<long long>::max(),
	                       int eng_id=0);
/*
Draw edges from the graph by negative sampling.

*/
void uniform_neg_sampling(const int* src_end_points,
                          const int* src_ind_ptr,
                          const int* target_indices,
                          int nnz,
                          int node_num,
                          int dst_node_num,
                          float neg_sample_scale,
                          bool replace,
                          int** dst_end_points,
                          int** dst_ind_ptr,
                          int** dst_edge_label,
                          int** dst_edge_count,
                          int* dst_nnz);

/*
Begin random walk from a given index
*/
void get_random_walk_nodes(const int* src_end_points,
                           const int* src_ind_ptr,
                           int nnz,
                           int node_num,
                           int initial_node,
                           int max_node_num,
                           int walk_length,
                           std::vector<int>* dst_indices);

/*
Randomly select the neighborhoods and merge
*/
void random_sel_neighbor_and_merge(const int* src_end_points,
                                   const int* src_ind_ptr,
                                   const int* src_node_ids,
                                   const int* sel_indices,
                                   int nnz,
                                   int sel_node_num,
                                   int neighbor_num,
                                   float neighbor_frac,
                                   bool sample_all,
                                   bool replace,
                                   std::vector<int>* dst_end_points,
                                   std::vector<int>* dst_ind_ptr,
                                   std::vector<int>* merged_node_ids,
                                   std::vector<int>* indices_in_merged);

private:
	RANDOM_ENGINE eng_[MAX_RANDOM_ENGINE_NUM];
};

void slice_csr_mat(const int* src_end_points,
                   const float* src_values,
                   const int* src_ind_ptr,
                   const int* src_row_ids,
                   const int* src_col_ids,
                   int src_row_num,
                   int src_col_num,
                   int src_nnz,
                   const int* sel_row_indices,
                   const int* sel_col_indices,
                   int dst_row_num,
                   int dst_col_num,
                   int** dst_end_points,
                   float** dst_values,
                   int** dst_ind_ptr,
                   int** dst_row_ids,
                   int** dst_col_ids,
                   int* dst_nnz);

} // namespace graph_sampler
#endif