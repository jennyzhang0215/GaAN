#include "Python.h"
#include "numpy/arrayobject.h"
#include "graph_sampler.h"
#include <sstream>
#include <iostream>
#include <string>
#include <utility>

#if PY_MAJOR_VERSION >= 3
static PyObject *GraphSamplerError;
static graph_sampler::GraphSampler handle;

#define CHECK_SEQUENCE(obj)                                                                                      \
{                                                                                                                \
  if (!PySequence_Check(obj)) {                                                                                  \
    PyErr_SetString(GraphSamplerError, "Need a sequence!");                                                      \
    return NULL;                                                                                                 \
  }                                                                                                              \
}

#define PY_CHECK_EQUAL(a, b)                                                                                     \
{                                                                                                                \
  if ((a) != (b)) {                                                                                              \
    std::ostringstream err_msg;                                                                                  \
    err_msg << "Line:" << __LINE__ << ", Check\"" << #a << " == " << #b << "\" failed";                             \
    PyErr_SetString(GraphSamplerError, err_msg.str().c_str());                                                   \
    return NULL;                                                                                                 \
  }                                                                                                              \
}                                                                                                                \


void alloc_npy_from_ptr(const int* arr_ptr, const size_t arr_size, PyObject** arr_obj) {
    npy_intp siz[] = { static_cast<npy_intp>(arr_size) };
    *arr_obj = PyArray_EMPTY(1, siz, NPY_INT, 0);
    memcpy(PyArray_DATA(*arr_obj), static_cast<const void*>(arr_ptr), sizeof(int) * arr_size);
    return;
}

void alloc_npy_from_ptr(const float* arr_ptr, const size_t arr_size, PyObject** arr_obj) {
    npy_intp siz[] = { static_cast<npy_intp>(arr_size) };
    *arr_obj = PyArray_EMPTY(1, siz, NPY_FLOAT32, 0);
    memcpy(PyArray_DATA(*arr_obj), static_cast<const void*>(arr_ptr), sizeof(float) * arr_size);
    return;
}

template<typename DType>
void alloc_npy_from_vector(const std::vector<DType> &arr_vec, PyObject** arr_obj) {
    alloc_npy_from_ptr(arr_vec.data(), arr_vec.size(), arr_obj);
    return;
}


/*
Inputs:
NDArray(int) src_end_points,
NDArray(int) src_ind_ptr,
NDArray(int) src_node_ids,
int src_undirected,
int initial_node,
int walk_length,
double return_prob,
int max_node_num,
int max_edge_num
---------------------------------
Outputs:
NDArray(int) subgraph_end_points,
NDArray(int) subgraph_ind_ptr,
NDArray(int) subgraph_node_ids
*/
static PyObject* random_walk(PyObject* self, PyObject* args) {
    PyObject* src_end_points;
    PyObject* src_ind_ptr;
    PyObject* src_node_ids;
    int src_undirected;
    int initial_node;
    int walk_length;
    double return_prob;
    int max_node_num;
    int max_edge_num;
    if (!PyArg_ParseTuple(args, "OOOiiidii",
                          &src_end_points,
                          &src_ind_ptr,
                          &src_node_ids,
                          &src_undirected,
                          &initial_node,
                          &walk_length,
                          &return_prob,
                          &max_node_num,
                          &max_edge_num)) return NULL;
    PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_node_ids), NPY_INT32);

    int src_node_num = PyArray_SIZE(src_ind_ptr) - 1;
    int src_edge_num = PyArray_SIZE(src_end_points);
    if (src_undirected) {
        src_edge_num /= 2;
    }
    std::vector<int> subgraph_end_points_vec;
    std::vector<int> subgraph_ind_ptr_vec;
    std::vector<int> subgraph_node_ids_vec;
    graph_sampler::SimpleGraph* subgraph = handle.random_walk(static_cast<int*>(PyArray_DATA(src_end_points)),
                                                              static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                                              static_cast<int*>(PyArray_DATA(src_node_ids)),
                                                              src_undirected,
                                                              src_node_num,
                                                              initial_node,
                                                              walk_length,
                                                              return_prob,
                                                              max_node_num,
                                                              max_edge_num);
    subgraph->convert_to_csr(&subgraph_end_points_vec, &subgraph_ind_ptr_vec, &subgraph_node_ids_vec,
                             static_cast<int*>(PyArray_DATA(src_node_ids)), src_node_num);
    PyObject* subgraph_end_points = NULL;
    PyObject* subgraph_ind_ptr = NULL;
    PyObject* subgraph_node_ids = NULL;
    alloc_npy_from_vector(subgraph_end_points_vec, &subgraph_end_points);
    alloc_npy_from_vector(subgraph_ind_ptr_vec, &subgraph_ind_ptr);
    alloc_npy_from_vector(subgraph_node_ids_vec, &subgraph_node_ids);
    delete subgraph;
    return Py_BuildValue("(NNN)", subgraph_end_points, subgraph_ind_ptr, subgraph_node_ids);
}

/*
Inputs:
NDArray(int) src_end_points,
NDArray(int) src_indptr,
NDArray(int) node_indices,
float neg_sample_scale,
int replace
---------------------------------
Outputs:
NDArray(int) end_points,
NDArray(int) indptr,
NDArray(int) edge_label
NDArray(int) edge_count
*/
static PyObject* uniform_neg_sampling(PyObject* self, PyObject* args) {
    PyObject* src_end_points;
    PyObject* src_ind_ptr;
    PyObject* node_indices;
    float neg_sample_scale;
    int replace;
    if (!PyArg_ParseTuple(args, "OOOfi",
                          &src_end_points,
                          &src_ind_ptr,
                          &node_indices,
                          &neg_sample_scale,
                          &replace)) return NULL;
    // Check Type
    PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(node_indices), NPY_INT32);

    int src_node_num = PyArray_SIZE(src_ind_ptr) - 1;
    int src_nnz = PyArray_SIZE(src_end_points);
    int dst_node_num = PyArray_SIZE(node_indices);
    int* dst_end_points_d = NULL;
    int* dst_indptr_d = NULL;
    int* dst_edge_label_d = NULL;
    int* dst_edge_count_d = NULL;
    int dst_nnz = 0;
    handle.uniform_neg_sampling(static_cast<int*>(PyArray_DATA(src_end_points)),
                                static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                static_cast<int*>(PyArray_DATA(node_indices)),
                                src_nnz,
                                src_node_num,
                                dst_node_num,
                                neg_sample_scale,
                                replace,
                                &dst_end_points_d,
                                &dst_indptr_d,
                                &dst_edge_label_d,
                                &dst_edge_count_d,
                                &dst_nnz);
    PyObject* dst_end_points = NULL;
    PyObject* dst_indptr = NULL;
    PyObject* dst_edge_label = NULL;
    PyObject* dst_edge_count = NULL;
    alloc_npy_from_ptr(dst_end_points_d, dst_nnz, &dst_end_points);
    alloc_npy_from_ptr(dst_indptr_d, dst_node_num + 1, &dst_indptr);
    alloc_npy_from_ptr(dst_edge_label_d, dst_nnz, &dst_edge_label);
    alloc_npy_from_ptr(dst_edge_count_d, dst_nnz, &dst_edge_count);
    delete[] dst_end_points_d;
    delete[] dst_indptr_d;
    delete[] dst_edge_label_d;
    delete[] dst_edge_count_d;
    return Py_BuildValue("(NNNN)", dst_end_points, dst_indptr, dst_edge_label, dst_edge_count);
}

/*
Inputs:
NDArray(int) src_end_points,
NDArray(int) src_indptr,
NDArray(int) src_node_ids,
NDArray(int) sel_indices,
int neighbor_num,
float neighbor_frac,
int sample_all,
int replace
---------------------------------
Outputs:
NDArray(int) dst_end_points,
NDArray(int) dst_ind_ptr,
NDArray(int) merged_node_ids
NDArray(int) indices_in_merged
*/
static PyObject* random_sel_neighbor_and_merge(PyObject* self, PyObject* args) {
    PyObject* src_end_points;
    PyObject* src_ind_ptr;
    PyObject* src_node_ids;
    PyObject* sel_indices;
    int neighbor_num;
    float neighbor_frac;
    int sample_all;
    int replace;
    if (!PyArg_ParseTuple(args, "OOOOifii",
                          &src_end_points,
                          &src_ind_ptr,
                          &src_node_ids,
                          &sel_indices,
                          &neighbor_num,
                          &neighbor_frac,
                          &sample_all,
                          &replace)) return NULL;
    // Check Type
    PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_node_ids), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(sel_indices), NPY_INT32);

    int src_node_num = PyArray_SIZE(src_ind_ptr) - 1;
    int src_nnz = PyArray_SIZE(src_end_points);
    int sel_node_num = PyArray_SIZE(sel_indices);
    std::vector<int> dst_end_points_vec;
    std::vector<int> dst_ind_ptr_vec;
    std::vector<int> merged_node_ids_vec;
    std::vector<int> indices_in_merged_vec;
    handle.random_sel_neighbor_and_merge(static_cast<int*>(PyArray_DATA(src_end_points)),
                                         static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                         static_cast<int*>(PyArray_DATA(src_node_ids)),
                                         static_cast<int*>(PyArray_DATA(sel_indices)),
                                         src_nnz,
                                         sel_node_num,
                                         neighbor_num,
                                         neighbor_frac,
                                         sample_all,
                                         replace,
                                         &dst_end_points_vec,
                                         &dst_ind_ptr_vec,
                                         &merged_node_ids_vec,
                                         &indices_in_merged_vec);
    PyObject* dst_end_points = NULL;
    PyObject* dst_ind_ptr = NULL;
    PyObject* merged_node_ids = NULL;
    PyObject* indices_in_merged = NULL;
    alloc_npy_from_vector(dst_end_points_vec, &dst_end_points);
    alloc_npy_from_vector(dst_ind_ptr_vec, &dst_ind_ptr);
    alloc_npy_from_vector(merged_node_ids_vec, &merged_node_ids);
    alloc_npy_from_vector(indices_in_merged_vec, &indices_in_merged);
    return Py_BuildValue("(NNNN)", dst_end_points, dst_ind_ptr, merged_node_ids, indices_in_merged);
}

/*
Inputs:
NDArray(int) src_end_points,
NDArray(int) src_indptr,
int initial_node,
int max_node_num,
int walk_length
---------------------------------
Outputs:
NDArray(int) dst_indices
*/
static PyObject* get_random_walk_nodes(PyObject* self, PyObject* args) {
  PyObject* src_end_points;
  PyObject* src_ind_ptr;
  int initial_node;
  int max_node_num;
  int walk_length;
  if (!PyArg_ParseTuple(args, "OOiii",
                        &src_end_points,
                        &src_ind_ptr,
                        &initial_node,
                        &max_node_num,
                        &walk_length)) return NULL;
  // Check Type
  PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
  PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
  int src_node_num = PyArray_SIZE(src_ind_ptr) - 1;
  int nnz = PyArray_SIZE(src_end_points);
  std::vector<int> dst_indices_vec;
  handle.get_random_walk_nodes(static_cast<int*>(PyArray_DATA(src_end_points)),
                               static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                               nnz,
                               src_node_num,
                               initial_node,
                               max_node_num,
                               walk_length,
                               &dst_indices_vec);
  PyObject* dst_indices = NULL;
  alloc_npy_from_vector(dst_indices_vec, &dst_indices);
  return Py_BuildValue("N", dst_indices);
}

/*
Inputs:
NDArray(int) seed
---------------------------------
Outputs:
NDArray(int) ret_val
*/
static PyObject* set_seed(PyObject* self, PyObject* args) {
    int seed;
    if (!PyArg_ParseTuple(args, "i", &seed)) return NULL;
    handle.set_seed(seed);
    return Py_BuildValue("i", 1);
}


/*
Inputs:
NDArray(int) src_end_points
NDArray(int) or None src_values
NDArray(int) src_ind_ptr
NDArray(int) src_row_ids
NDArray(int) src_col_ids
NDArray(int) sel_row_indices
NDArray(int) sel_col_indices
---------------------------------
Outputs:
NDArray(int) dst_end_points
NDArray(int) dst_ind_ptr
NDArray(int) dst_row_ids
NDArray(int) dst_col_ids
int dst_row_num
int dst_col_num
int dst_nnz
*/
static PyObject* csr_submat(PyObject* self, PyObject* args) {
    PyObject* src_end_points;
    PyObject* src_values;
    PyObject* src_ind_ptr;
    PyObject* src_row_ids;
    PyObject* src_col_ids;
    PyObject* sel_row_indices;
    PyObject* sel_col_indices;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &src_end_points,
                          &src_values,
                          &src_ind_ptr,
                          &src_row_ids,
                          &src_col_ids,
                          &sel_row_indices,
                          &sel_col_indices)) return NULL;
    // Check Type
    PY_CHECK_EQUAL(PyArray_TYPE(src_end_points), NPY_INT32);
    if(src_values != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(src_values), NPY_FLOAT32);
    }
    PY_CHECK_EQUAL(PyArray_TYPE(src_ind_ptr), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_row_ids), NPY_INT32);
    PY_CHECK_EQUAL(PyArray_TYPE(src_col_ids), NPY_INT32);
    if(sel_row_indices != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(sel_row_indices), NPY_INT32);
    }
    if(sel_col_indices != Py_None) {
      PY_CHECK_EQUAL(PyArray_TYPE(sel_col_indices), NPY_INT32);
    }


    long long src_row_num = PyArray_SIZE(src_row_ids);
    long long src_col_num = PyArray_SIZE(src_col_ids);
    long long src_nnz = PyArray_SIZE(src_end_points);
    ASSERT(src_row_num <= std::numeric_limits<int>::max());
    ASSERT(src_col_num <= std::numeric_limits<int>::max());
    ASSERT(src_nnz < std::numeric_limits<int>::max());
    int dst_row_num = (sel_row_indices == Py_None) ? src_row_num : PyArray_SIZE(sel_row_indices);
    int dst_col_num = (sel_col_indices == Py_None) ? src_col_num : PyArray_SIZE(sel_col_indices);
    float* src_values_ptr = (src_values == Py_None) ? nullptr : static_cast<float*>(PyArray_DATA(src_values));
    int* sel_row_indices_ptr = (sel_row_indices == Py_None) ? nullptr : static_cast<int*>(PyArray_DATA(sel_row_indices));
    int* sel_col_indices_ptr = (sel_col_indices == Py_None) ? nullptr : static_cast<int*>(PyArray_DATA(sel_col_indices));
    int* dst_end_points_d = NULL;
    float* dst_values_d = NULL;
    int* dst_ind_ptr_d = NULL;
    int* dst_row_ids_d = NULL;
    int* dst_col_ids_d = NULL;
    int dst_nnz;
    graph_sampler::slice_csr_mat(static_cast<int*>(PyArray_DATA(src_end_points)),
                                 src_values_ptr,
                                 static_cast<int*>(PyArray_DATA(src_ind_ptr)),
                                 static_cast<int*>(PyArray_DATA(src_row_ids)),
                                 static_cast<int*>(PyArray_DATA(src_col_ids)),
                                 src_row_num,
                                 src_col_num,
                                 src_nnz,
                                 sel_row_indices_ptr,
                                 sel_col_indices_ptr,
                                 dst_row_num,
                                 dst_col_num,
                                 &dst_end_points_d,
                                 &dst_values_d,
                                 &dst_ind_ptr_d,
                                 &dst_row_ids_d,
                                 &dst_col_ids_d,
                                 &dst_nnz);
    PyObject* dst_end_points = NULL;
    PyObject* dst_values = NULL;
    PyObject* dst_ind_ptr = NULL;
    PyObject* dst_row_ids = NULL;
    PyObject* dst_col_ids = NULL;
    alloc_npy_from_ptr(dst_end_points_d, dst_nnz, &dst_end_points);
    if(dst_values_d == nullptr) {
        Py_INCREF(Py_None);
        dst_values = Py_None;
    } else {
        alloc_npy_from_ptr(dst_values_d, dst_nnz, &dst_values);
    }
    alloc_npy_from_ptr(dst_ind_ptr_d, dst_row_num + 1, &dst_ind_ptr);
    alloc_npy_from_ptr(dst_row_ids_d, dst_row_num, &dst_row_ids);
    alloc_npy_from_ptr(dst_col_ids_d, dst_col_num, &dst_col_ids);

    //Clear Allocated Variables
    delete[] dst_end_points_d;
    if (dst_values_d != nullptr) delete[] dst_values_d;
    delete[] dst_ind_ptr_d;
    delete[] dst_row_ids_d;
    delete[] dst_col_ids_d;
    return Py_BuildValue("(NNNNN)", dst_end_points, dst_values, dst_ind_ptr, dst_row_ids, dst_col_ids);
}


static PyMethodDef myextension_methods[] = {
    {"random_walk", (PyCFunction)random_walk, METH_VARARGS, NULL},
    {"uniform_neg_sampling", (PyCFunction)uniform_neg_sampling, METH_VARARGS, NULL},
    {"random_sel_neighbor_and_merge", (PyCFunction)random_sel_neighbor_and_merge, METH_VARARGS, NULL},
    {"get_random_walk_nodes", (PyCFunction)get_random_walk_nodes, METH_VARARGS, NULL},
    {"set_seed", (PyCFunction)set_seed, METH_VARARGS, NULL},
    {"csr_submat", (PyCFunction)csr_submat, METH_VARARGS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_graph_sampler",
        NULL,
        -1,
        myextension_methods
};


PyMODINIT_FUNC
PyInit__graph_sampler(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
      return NULL;
    import_array();
    GraphSamplerError = PyErr_NewException("graph_sampler.error", NULL, NULL);
    Py_INCREF(GraphSamplerError);
    PyModule_AddObject(m, "graph_sampler.error", GraphSamplerError);
    return m;
}
#endif

