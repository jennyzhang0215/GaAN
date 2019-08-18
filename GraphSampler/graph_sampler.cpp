#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <ctime>
#include "graph_sampler.h"

//#define MXGRAPH_OMP_THREAD_NUM 8

int mxgraph_set_omp_thread_num() {
  int omp_thread_num_used = std::min(omp_get_max_threads(), 16);
  omp_set_num_threads(omp_thread_num_used);
  return omp_thread_num_used;
}

namespace graph_sampler {

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
                   int* dst_nnz) {
    ASSERT(dst_row_num > 0);
    ASSERT(dst_col_num > 0);
    if(sel_row_indices == nullptr) {
        ASSERT(dst_row_num == src_row_num);
    }
    if (sel_col_indices == nullptr) {
        ASSERT(dst_col_num == src_col_num);
    }

    if(sel_row_indices == nullptr && sel_col_indices == nullptr) {
        //Handle the special case where we could copy the source
        *dst_row_ids = new int[src_row_num];
        *dst_col_ids = new int[src_col_num];
        *dst_end_points = new int[src_nnz];
        *dst_values = new float[src_nnz];
        *dst_ind_ptr = new int[src_row_num + 1];
        memcpy(*dst_row_ids, src_row_ids, sizeof(int) * src_row_num);
        memcpy(*dst_col_ids, src_col_ids, sizeof(int) * src_col_num);
        memcpy(*dst_end_points, src_end_points, sizeof(int) * src_nnz);
        if(src_values != nullptr) memcpy(*dst_values, src_values, sizeof(float) * src_nnz);
        memcpy(*dst_ind_ptr, src_ind_ptr, sizeof(int) * (src_row_num + 1));
        return;
    }

    *dst_row_ids = new int[dst_row_num];
    *dst_col_ids = new int[dst_col_num];
    if(sel_row_indices == nullptr) {
        memcpy(*dst_row_ids, src_row_ids, sizeof(int) * src_row_num);
    } else {
        for (int i = 0; i < dst_row_num; i++) {
            (*dst_row_ids)[i] = src_row_ids[sel_row_indices[i]];
        }
    }
    if(sel_col_indices == nullptr) {
        // If all columns are selected, we can accelerate the computation
        memcpy(*dst_col_ids, src_col_ids, sizeof(int) * src_col_num);
        *dst_ind_ptr = new int[dst_row_num + 1];
        *dst_nnz = 0;
        (*dst_ind_ptr)[0] = 0;
        for(int i = 0; i < dst_row_num; i++) {
            int ele_num = src_ind_ptr[sel_row_indices[i] + 1] - src_ind_ptr[sel_row_indices[i]];
            (*dst_ind_ptr)[i + 1] = (*dst_ind_ptr)[i] + ele_num;
            *dst_nnz += ele_num;
        }
        *dst_end_points = new int[*dst_nnz];
        if (src_values != nullptr) *dst_values = new float[*dst_nnz];
        for(int i = 0; i < dst_row_num; i++) {
            int ele_num = src_ind_ptr[sel_row_indices[i] + 1] - src_ind_ptr[sel_row_indices[i]];
            memcpy((*dst_end_points) + (*dst_ind_ptr)[i], src_end_points + src_ind_ptr[sel_row_indices[i]], sizeof(int) * ele_num);
            if (src_values != nullptr) memcpy((*dst_values) + (*dst_ind_ptr)[i], src_values + src_ind_ptr[sel_row_indices[i]], sizeof(float) * ele_num);
        }
        return;
    } else {
        for (int i = 0; i < dst_col_num; i++) {
            (*dst_col_ids)[i] = src_col_ids[sel_col_indices[i]];
        }
    }
    std::unordered_map<int, int> col_idx_map;
    std::vector<std::vector<int> > vec_end_points(dst_row_num);
    std::vector<std::vector<float> > vec_values(dst_row_num);
    for(int i = 0; i < dst_col_num; i++) {
        col_idx_map.insert(std::make_pair(sel_col_indices[i], i));
    }
    ASSERT(col_idx_map.size() == dst_col_num);
    int  global_nnz = 0;
    int local_nnz = 0;
#pragma omp parallel for private(local_nnz) reduction(+:global_nnz)
    for(int i = 0; i < dst_row_num; i++) {
        local_nnz = 0;
        int idx = (sel_row_indices == nullptr) ? i : sel_row_indices[i];
        for(int j = src_ind_ptr[idx]; j < src_ind_ptr[idx + 1]; j++) {
            std::unordered_map<int, int>::iterator it = col_idx_map.find(src_end_points[j]);
            if (it != col_idx_map.end()) {
                if (src_values != nullptr) vec_values[i].push_back(src_values[j]);
                vec_end_points[i].push_back(it->second);
                local_nnz++;
            }
        }
        global_nnz += local_nnz;
    }
    *dst_nnz = global_nnz;
    ASSERT(*dst_nnz > 0);
    *dst_end_points = new int[*dst_nnz];
    if (src_values != nullptr) *dst_values = new float[*dst_nnz];
    *dst_ind_ptr = new int[dst_row_num + 1];
    int shift = 0;
    for(int i = 0; i < dst_row_num; i++) {
        (*dst_ind_ptr)[i] = shift;
        shift += vec_end_points[i].size();
    }
    (*dst_ind_ptr)[dst_row_num] = shift;
    for(int i = 0; i < dst_row_num; i++) {
        memcpy((*dst_end_points) + (*dst_ind_ptr)[i],
               static_cast<const void*>(vec_end_points[i].data()),
               sizeof(int) * vec_end_points[i].size());
        if (src_values != nullptr) {
            memcpy((*dst_values) + (*dst_ind_ptr)[i],
                    static_cast<const void*>(vec_values[i].data()),
                    sizeof(float) * vec_values[i].size());
        }
    }
    return;
}

bool check_subgraph(const SimpleGraph& graph,
                    const std::vector<int>& end_points,
                    const std::vector<int>& ind_ptr,
                    const std::vector<int>& node_ids) {
    GRAPH_DATA_T::const_iterator it;
    for(int i=0; i< graph.node_num(); i++) {
        it = graph.data()->find(node_ids[i]);
        if (it == graph.data()->end()) return false;
        for(int j=ind_ptr[i]; j < ind_ptr[i+1]; j++) {
            std::unordered_set<int>::const_iterator eit = it->second.find(node_ids[end_points[j]]);
            if (eit == it->second.end()) return false;
        }
    }
    return true;
}

bool check_equal(const SimpleGraph& graph,
                 const std::vector<int>& end_points,
                 const std::vector<int>& ind_ptr,
                 const std::vector<int>& node_ids) {
    if(node_ids.size() != graph.node_num()) {
        return false;
    }
    if(ind_ptr.size() != graph.node_num() + 1) {
        return false;
    }
    if(graph.undirected()) {
        if (end_points.size() != graph.edge_num() * 2) return false;
    } else {
        if (end_points.size() != graph.edge_num()) return false;
    }
    return check_subgraph(graph, end_points, ind_ptr, node_ids);
}

/* One step of the classic random walk
At every step, we will return to the original node with return_p. Otherwise, we will jump randomly to a conneted node.
See [KDD06] Sampling from Large Graphs
 */
std::pair<int, bool> step_random_walk(int current_node_id,
                                      int original_node_id,
                                      const int* end_points,
                                      const int* ind_ptr,
                                      RANDOM_ENGINE* gen,
                                      double return_prob=0.15) {
    std::bernoulli_distribution dis_return(return_prob);
    bool is_return = dis_return(*gen);
	if(is_return) {
		return std::make_pair(original_node_id, is_return);
	} else {
		std::uniform_int_distribution<> dis(ind_ptr[current_node_id], ind_ptr[current_node_id + 1] - 1);
		return std::make_pair(end_points[dis(*gen)], is_return);
	}
}

SimpleGraph* GraphSampler::random_walk(const int* src_end_points,
                                       const int* src_ind_ptr,
                                       const int* src_node_ids,
                                       bool src_undirected,
                                       int src_node_num,
                                       int initial_node,
                                       int walk_length,
                                       double return_prob,
                                       int max_node_num,
                                       long long max_edge_num,
                                       int eng_id) {
    std::cout << "return_prob=" << return_prob << std::endl;
    std::cout << "walk length=" << walk_length << std::endl;
    ASSERT(return_prob >= 0 && return_prob <= 1);
    SimpleGraph* dst_graph = new SimpleGraph(src_undirected, max_node_num, max_edge_num);
    if (initial_node < 0) {
        std::uniform_int_distribution<int> dis(0, src_node_num);
        initial_node = dis(eng_[eng_id]);
    }
    int old_node = initial_node;
    for (int j = 0; j < walk_length; j++) {
        std::pair<int, bool> new_sample = step_random_walk(old_node, initial_node,
                                                           src_end_points, src_ind_ptr,
                                                           &eng_[eng_id], return_prob);
        if(!new_sample.second) {
            if (!dst_graph->insert_new_edge(std::make_pair(old_node, new_sample.first))) break;
        }
        old_node = new_sample.first;
    }
    return dst_graph;
}

void choice_with_exist_number(std::vector<int> *sampled_value,
                              std::vector<int> *count,
                              const int* exist_number,
                              int C, int N, int K,
                              bool replace,
                              RANDOM_ENGINE* gen) {
  ASSERT(C < N);
  std::unordered_map<int, int> pool;
  std::unordered_map<int, int> pos_of_value;
  std::unordered_map<int, int>::iterator it;
  // 1. Initialize the pool
  for(int i = 0; i < C; i++) {
    int val = exist_number[i];
    if (val == i) continue;
    // Get val_pos
    int val_pos = val;
    it = pos_of_value.find(val);
    if(it != pos_of_value.end()) {
      val_pos = it->second;
    }
    // Get ith_val
    int ith_val = i;
    it = pool.find(ith_val);
    if(it != pool.end()) {
      ith_val = it->second;
    }
    pos_of_value[val] = i;
    pos_of_value[ith_val] = val_pos;
    pool[val_pos] = ith_val;
    pool[i] = val;
  }
  // 2. Draw samples
  int lower = C;
  if(replace) {
    // 2.1 If replace, we sample K values according to the same distribution
    std::unordered_map<int, int> ret_hash_map;
    std::uniform_int_distribution<int> dis(lower, N - 1);
    for (int i = 0; i < K; i++) {
      int val = dis(*gen);
      it = pool.find(val);
      if(it != pool.end()) {
        val = it->second;
      }
      it = ret_hash_map.find(val);
      if(it != ret_hash_map.end()) {
        it->second += 1;
      } else {
        ret_hash_map[val] = 1;
      }
    }
    for(const auto&ele: ret_hash_map) {
      sampled_value->push_back(ele.first);
      count->push_back(ele.second);
    }
  } else {
    // 2.2 If non replacement, we use the method described in
    // https://codegolf.stackexchange.com/questions/4772/random-sampling-without-replacement
    if(lower + K >= N) {
      for(int i=lower; i < N; i++) {
        int val = i;
        it = pool.find(val);
        if(it != pool.end()) {
          val = it->second;
        }
        sampled_value->push_back(val);
        count->push_back(1);
      }
    } else {
      for(int i=0; i < K; i++) {
        if (lower >= N) break;
        std::uniform_int_distribution<int> dis(lower, N - 1);
        int sample_val = dis(*gen);
        it = pool.find(sample_val);
        if(it != pool.end()) {
          sampled_value->push_back(it->second);
        } else {
          sampled_value->push_back(sample_val);
        }
        it = pool.find(lower);
        if(it != pool.end()) {
          pool[sample_val] = it->second;
        } else {
          pool[sample_val] = lower;
        }
        count->push_back(1);
        lower += 1;
      }
    }
  }
}

void GraphSampler::uniform_neg_sampling(const int* src_end_points,
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
                                        int* dst_nnz) {
  std::vector<std::vector<int> > end_points_vec(dst_node_num);
  std::vector<std::vector<int> > edge_label_vec(dst_node_num);
  std::vector<std::vector<int> > edge_count_vec(dst_node_num);
  *dst_ind_ptr = new int[dst_node_num + 1];
  memset(*dst_ind_ptr, 0, sizeof(int) * (dst_node_num + 1));
  int global_nnz = 0;
  std::clock_t start;
  #pragma omp parallel for reduction(+:global_nnz)
  for(int i=0; i < dst_node_num; i++) {
    int ind = target_indices[i];
    int tid = omp_get_thread_num();
    int p_begin = src_ind_ptr[ind];
    int p_end = src_ind_ptr[ind + 1];
    int pos_num = p_end - p_begin;
    if(pos_num == 0) continue;
    ASSERT(p_begin >= 0 && p_end <= nnz);
    int neg_sample_num = std::min(static_cast<int>(std::ceil(pos_num * neg_sample_scale)),
                                  node_num);
    // Insert the positive edges
    end_points_vec[i].insert(end_points_vec[i].end(),
                             src_end_points + p_begin,
                             src_end_points + p_end);
    edge_label_vec[i].insert(edge_label_vec[i].end(), pos_num, 1);
    edge_count_vec[i].insert(edge_count_vec[i].end(), pos_num, 1);
    // Insert the negative edges
    choice_with_exist_number(&end_points_vec[i], &edge_count_vec[i],
                             src_end_points + p_begin, pos_num, node_num,
                             neg_sample_num, replace, &this->eng_[tid]);
    edge_label_vec[i].insert(edge_label_vec[i].end(),
                              end_points_vec[i].size() - pos_num, -1);
    global_nnz += end_points_vec[i].size();
  }
  *dst_nnz = global_nnz;
  ASSERT(global_nnz > 0);
  *dst_end_points = new int[global_nnz];
  *dst_edge_label = new int[global_nnz];
  *dst_edge_count = new int[global_nnz];
  int curr_ind = 0;
  (*dst_ind_ptr)[0] = 0;
  for(int i=0; i < dst_node_num; i++) {
    int curr_size = end_points_vec[i].size();
    if(curr_size > 0) {
      std::memcpy((*dst_end_points) + curr_ind,
                  static_cast<const void*>(end_points_vec[i].data()),
                  sizeof(int) * curr_size);
      std::memcpy((*dst_edge_label) + curr_ind,
                  static_cast<const void*>(edge_label_vec[i].data()),
                  sizeof(int) * curr_size);
      std::memcpy((*dst_edge_count) + curr_ind,
                  static_cast<const void*>(edge_count_vec[i].data()),
                  sizeof(int) * curr_size);
    }
    (*dst_ind_ptr)[i + 1] = curr_ind + curr_size;
    curr_ind += curr_size;
  }
}

void GraphSampler::get_random_walk_nodes(const int* src_end_points,
                                         const int* src_ind_ptr,
                                         int nnz,
                                         int node_num,
                                         int initial_node,
                                         int max_node_num,
                                         int walk_length,
                                         std::vector<int>* dst_indices) {
  std::unordered_set<int> indices;
  indices.insert(initial_node);
  int curr_ind = initial_node;
  for(int i = 0; i < walk_length; i++) {
    int p_begin = src_ind_ptr[curr_ind];
    int p_end = src_ind_ptr[curr_ind + 1];
    ASSERT(p_end >= p_begin);
    if(p_end == p_begin) {
      break;
    }
    std::uniform_int_distribution<int> dis(p_begin, p_end - 1);
    curr_ind = src_end_points[dis(this->eng_[0])];
    indices.insert(curr_ind);
    if(indices.size() >= max_node_num) {
      break;
    }
  }
  *dst_indices = std::vector<int>(indices.begin(), indices.end());
  return;
}

void uniform_choice_set(int* dst, const int* src, int p_begin, int p_end, int num, bool replace, RANDOM_ENGINE* gen) {
  if(replace) {
    std::uniform_int_distribution<int> dis(p_begin, p_end - 1);
    for(int i = 0; i < num; i++) {
      dst[i] = src[dis(*gen)];
    }
  } else {
    std::unordered_map<int, int> pool;
    std::unordered_map<int, int>::iterator it;
    for(int lower = 0; lower < num; lower++) {
      std::uniform_int_distribution<int> dis(lower, num - 1);
      int sample_val = dis(*gen);
      it = pool.find(sample_val);
      if (it != pool.end()) {
        dst[lower] = src[it->second + p_begin];
      }
      else {
        dst[lower] = src[sample_val + p_begin];
      }
      it = pool.find(lower);
      if (it != pool.end()) {
        pool[sample_val] = it->second;
      }
      else {
        pool[sample_val] = lower;
      }
    }
  }
}

void GraphSampler::random_sel_neighbor_and_merge(const int* src_end_points,
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
                                                 std::vector<int>* indices_in_merged) {
  std::unordered_map<int, int> merged_node_id_map;  // We can actually use a vector with length=node_num as the hash_map
  *dst_ind_ptr = std::vector<int>(sel_node_num + 1);
  *indices_in_merged = std::vector<int>(sel_node_num, -1);
  (*dst_ind_ptr)[0] = 0;
  // Fill in the indptr, get the dst_nnz
  for(int i = 0; i < sel_node_num; i++) {
    int ind = sel_indices[i];
    merged_node_id_map.insert(std::make_pair(src_node_ids[ind], 0));
    int p_begin = src_ind_ptr[ind];
    int p_end = src_ind_ptr[ind + 1];
    int sample_num = 0;
    if(sample_all) {
      sample_num = p_end - p_begin;
    } else {
      if (neighbor_frac > 0.0) {
        ASSERT(neighbor_num < 0);
        sample_num = static_cast<int>(std::round(static_cast<float>(p_end - p_begin) * neighbor_frac));
        sample_num = std::max(sample_num, 15);
      } else {
        sample_num = neighbor_num;
      }
      sample_num = std::min(sample_num, p_end - p_begin);
    }
    (*dst_ind_ptr)[i + 1] = sample_num + (*dst_ind_ptr)[i];
  }
  int dst_nnz = (*dst_ind_ptr)[sel_node_num];
  ASSERT(dst_nnz >= 0);
  // Fill the end_points
  *dst_end_points = std::vector<int>(dst_nnz);
  int omp_thread_num_used = mxgraph_set_omp_thread_num();
  std::vector<std::unordered_map<int, int> > thread_map_vec(omp_thread_num_used);
  if(sample_all) {
#pragma omp parallel for
    for(int i = 0; i < sel_node_num; i++) {
      int tid = omp_get_thread_num();
      int ind = sel_indices[i];
      int p_begin = src_ind_ptr[ind];
      int p_end = src_ind_ptr[ind + 1];
      int shift = (*dst_ind_ptr)[i];
      // memcpy(dst_end_points->data() + shift, src_end_points + p_begin, sizeof(int) * (p_end - p_begin));
      for(int j = p_begin; j < p_end; j++) {
        *(dst_end_points->data() + shift + j - p_begin) = *(src_end_points + j);
        thread_map_vec[tid].insert(std::make_pair(src_node_ids[src_end_points[j]], -1));
      }
    }
  } else {
    // Calculate the end_points
#pragma omp parallel for
    for (int i = 0; i < sel_node_num; i++) {
      int tid = omp_get_thread_num();
      int ind = sel_indices[i];
      int p_begin = src_ind_ptr[ind];
      int p_end = src_ind_ptr[ind + 1];
      int shift = (*dst_ind_ptr)[i];
      int sample_num = (*dst_ind_ptr)[i + 1] - (*dst_ind_ptr)[i];
      int curr_size = dst_end_points->size();
      uniform_choice_set(dst_end_points->data() + shift, src_end_points, p_begin, p_end, sample_num, replace, &this->eng_[0]);
      for (int j = 0; j < sample_num; j++) {
        thread_map_vec[tid].insert(std::make_pair(src_node_ids[*(dst_end_points->data() + shift + j)], -1));
      }
    }
  }
  for(int i = 0; i < omp_thread_num_used; i++) {
    merged_node_id_map.insert(thread_map_vec[i].begin(), thread_map_vec[i].end());
  }
  // Map the id to the indices in the new vector 
  int counter = 0;
  for(std::unordered_map<int, int>::iterator it = merged_node_id_map.begin(); it != merged_node_id_map.end(); ++it) {
    merged_node_ids->push_back(it->first);
    it->second = counter;
    counter += 1;
  }
#pragma omp parallel for
  for(int i = 0; i < sel_node_num; i++) {
    (*indices_in_merged)[i] = merged_node_id_map[src_node_ids[sel_indices[i]]];
  }
#pragma omp parallel for
  for(int i = 0; i < dst_end_points->size(); i++) {
    (*dst_end_points)[i] = merged_node_id_map[src_node_ids[(*dst_end_points)[i]]];
  }
}

} // namespace graph_sampler

int main() {
    using namespace graph_sampler;
    int src_end_points[] = {1, 2, 5,
                        0, 2, 4, 5,
                        0, 1, 3, 4,
                        2, 4, 7,
                        1, 2, 3,
                        0, 1, 6,
                        5, 7,
                        3, 6};
    int src_ind_ptr[] = {0, 3, 7, 11, 14, 17, 20, 22, 24, 26};
    int src_node_ids[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int src_node_num = 8;
    int initial_node = 1;
    int walk_length = 10;
    double return_prob = 0.15;
    int max_node_num = 100;
    int max_edge_num = 100;
    GraphSampler handle;
    handle.set_seed(10);
    SimpleGraph* subgraph = handle.random_walk(src_end_points,
                                                              src_ind_ptr,
                                                              src_node_ids,
                                                              true,
                                                              src_node_num,
                                                              initial_node,
                                                              walk_length,
                                                              return_prob,
                                                              max_node_num,
                                                              max_edge_num);
    std::cout << "subgraph node_num=" << subgraph->node_num() << " edge_num=" << subgraph->edge_num();
    for(const auto &ele: *(subgraph->data())) {
        std::cout << ele.first << "->";
        for(int node : ele.second) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }
    std::vector<int> subgraph_end_points;
    std::vector<int> subgraph_ind_ptr;
    std::vector<int> subgraph_node_ids;
    subgraph->convert_to_csr(&subgraph_end_points, &subgraph_ind_ptr, &subgraph_node_ids, src_node_ids, src_node_num);
    std::cout << "Subgraph:" << std::endl;
    std::cout << "end_points:";
    for (int node: subgraph_end_points) {
        std::cout << " " << node;
    }
    std::cout << std::endl;
    std::cout << "ind_ptr:";
    for (int node: subgraph_ind_ptr) {
        std::cout << " " << node;
    }
    std::cout << std::endl;
    std::cout << "node_ids:";
    for (int node: subgraph_node_ids) {
        std::cout << " " << node;
    }
    std::cout << std::endl;
    std::cout << "Check consistency...";
    ASSERT(check_equal(*subgraph, subgraph_end_points, subgraph_ind_ptr, subgraph_node_ids));
    std::cout << "Success!" << std::endl;
    delete subgraph;
}