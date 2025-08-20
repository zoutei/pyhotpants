// hotpants_ext.c
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

// Include the original hotpants headers
#include "globals.h"
#include "functions.h"

// Structure to hold hotpants configuration from Python
typedef struct {
    float tuthresh, tlthresh, tuktresh, tgain, trdnoise, tpedestal;
    float iuthresh, ilthresh, iuktresh, igain, irdnoise, ipedestal;
    int rkernel;
    int ko, bgo;
    float fitthresh, scale_fitthresh, min_frac_stamps, ks, kfm;
    int nss, rss;
    int nregx, nregy, nstampx, nstampy;
    char force_convolve[2], normalize[2], fom[2];
    float fillval, fillval_noise;
    int verbose;
    int rescale_ok, conv_var, use_pca;
    int ngauss;
    int *deg_fixe;
    float *sigma_gauss;
    int fwkernel, fwksstamp, ncomp_ker, ncomp, n_bg_vectors, n_comp_total;
    float stat_sig, kf_spread_mask1;
} hotpants_config_t;

// Helper function declarations
static int parse_config_dict(PyObject *config_dict, hotpants_config_t *config);
static void init_hotpants_state(hotpants_state_t *state, int nx, int ny, const hotpants_config_t *config);
static void free_hotpants_state(hotpants_state_t *state);

// Python function wrappers
static PyObject *py_make_input_mask(PyObject *self, PyObject *args) {
    PyArrayObject *template_arr, *image_arr;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &template_arr, &PyArray_Type, &image_arr, &PyDict_Type, &config_dict_obj)) {
        return NULL;
    }
    int ny = (int)PyArray_DIM(template_arr, 0);
    int nx = (int)PyArray_DIM(template_arr, 1);
    hotpants_config_t config;
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;

    hotpants_state_t state;
    init_hotpants_state(&state, nx, ny, &config);
    
    makeInputMask(&state, (float*)PyArray_DATA(template_arr), (float*)PyArray_DATA(image_arr), state.mRData);
    
    npy_intp dims[2] = {ny, nx};
    PyObject* mask_arr = PyArray_SimpleNewFromData(2, dims, NPY_INT32, state.mRData);
    Py_INCREF(mask_arr);
    
    free_hotpants_state(&state);
    return mask_arr;
}

static PyObject *py_calculate_noise_image(PyObject *self, PyObject *args) {
    PyArrayObject *image_arr;
    PyObject *config_dict_obj;
    int is_template;

    if (!PyArg_ParseTuple(args, "O!O!p", &PyArray_Type, &image_arr, &PyDict_Type, &config_dict_obj, &is_template)) {
        return NULL;
    }
    int ny = (int)PyArray_DIM(image_arr, 0);
    int nx = (int)PyArray_DIM(image_arr, 1);
    hotpants_config_t config;
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state;
    init_hotpants_state(&state, nx, ny, &config);
    
    float* data = (float*)PyArray_DATA(image_arr);
    float inv_gain = is_template ? 1.0 / state.tGain : 1.0 / state.iGain;
    float rdnoise = is_template ? state.tRdnoise / state.tGain : state.iRdnoise / state.iGain;
    
    float *noise_data = makeNoiseImage4(&state, data, inv_gain, rdnoise);
    
    npy_intp dims[2] = {ny, nx};
    PyObject* noise_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, noise_data);
    Py_INCREF(noise_arr);
    
    free_hotpants_state(&state);
    return noise_arr;
}

static PyObject *py_find_stamps(PyObject *self, PyObject *args) {
    PyArrayObject *template_arr, *image_arr, *input_mask_arr;
    double fit_thresh;
    PyObject *config_dict_obj;
    
    if (!PyArg_ParseTuple(args, "O!O!O!dO!", &PyArray_Type, &template_arr, &PyArray_Type, &image_arr, &PyArray_Type, &input_mask_arr, &fit_thresh, &PyDict_Type, &config_dict_obj)) {
        return NULL;
    }
    int ny = (int)PyArray_DIM(template_arr, 0);
    int nx = (int)PyArray_DIM(template_arr, 1);
    hotpants_config_t config;
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state;
    init_hotpants_state(&state, nx, ny, &config);
    state.mRData = (int*)PyArray_DATA(input_mask_arr);
    state.kerFitThresh = fit_thresh;
    
    int niS = 0, ntS = 0;
    int rXMin = 0, rYMin = 0, rXMax = nx - 1, rYMax = ny - 1;
    
    for (int l = 0; l < state.nStampY; l++) {
        for (int k = 0; k < state.nStampX; k++) {
            int sXMin = rXMin + (int)(k * (rXMax - rXMin + 1.0) / state.nStampX);
            int sYMin = rYMin + (int)(l * (rYMax - rYMin + 1.0) / state.nStampY);
            int sXMax = imin(sXMin + state.fwStamp - 1, rXMax);
            int sYMax = imin(sYMin + state.fwStamp - 1, rYMax);
            
            state.tStamps[ntS].sscnt = state.tStamps[ntS].nss = 0; state.tStamps[ntS].chi2 = 0.0;
            state.iStamps[niS].sscnt = state.iStamps[niS].nss = 0; state.iStamps[niS].chi2 = 0.0;
            
            buildStamps(&state, sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 1, rXMin, rYMin,
                        state.iStamps, state.tStamps, (float*)PyArray_DATA(image_arr),
                        (float*)PyArray_DATA(template_arr), 0.0, 0.0);
            
            if (!(strncmp(state.forceConvolve_str, "i", 1)==0)) { if (state.tStamps[ntS].nss > 0) ntS += 1; }
            if (!(strncmp(state.forceConvolve_str, "t", 1)==0)) { if (state.iStamps[niS].nss > 0) niS += 1; }
        }
    }
    
    PyObject* stamp_list = PyList_New(0);
    if (!stamp_list) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to create Python list"); return NULL; }

    if (!(strncmp(state.forceConvolve_str, "i", 1)==0)) {
        for(int i = 0; i < ntS; ++i) {
            for(int j=0; j<state.tStamps[i].nss; j++) {
                PyObject* stamp_tuple = PyTuple_New(2);
                PyTuple_SetItem(stamp_tuple, 0, PyFloat_FromDouble(state.tStamps[i].xss[j]));
                PyTuple_SetItem(stamp_tuple, 1, PyFloat_FromDouble(state.tStamps[i].yss[j]));
                PyList_Append(stamp_list, stamp_tuple);
            }
        }
    }
    if (!(strncmp(state.forceConvolve_str, "t", 1)==0)) {
        for(int i = 0; i < niS; ++i) {
            for(int j=0; j<state.iStamps[i].nss; j++) {
                PyObject* stamp_tuple = PyTuple_New(2);
                PyTuple_SetItem(stamp_tuple, 0, PyFloat_FromDouble(state.iStamps[i].xss[j]));
                PyTuple_SetItem(stamp_tuple, 1, PyFloat_FromDouble(state.iStamps[i].yss[j]));
                PyList_Append(stamp_list, stamp_tuple);
            }
        }
    }
    free_hotpants_state(&state);
    return stamp_list;
}

static PyObject *py_fit_stamps_and_get_fom(PyObject *self, PyObject *args) {
    PyArrayObject *conv_arr, *ref_arr, *noise_arr;
    PyObject *stamps_list, *config_dict_obj;
    char* conv_dir;
    
    if (!PyArg_ParseTuple(args, "O!O!O!sO!", &PyArray_Type, &conv_arr, &PyArray_Type, &ref_arr, &PyArray_Type, &noise_arr, &conv_dir, &PyList_Type, &stamps_list, &PyDict_Type, &config_dict_obj)) {
        return NULL;
    }
    int ny = (int)PyArray_DIM(conv_arr, 0); int nx = (int)PyArray_DIM(conv_arr, 1);
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);
    
    state.mRData = (int*)calloc(nx*ny, sizeof(int)); if(!state.mRData) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate mask"); return NULL; }
    
    state.nS = PyList_Size(stamps_list);
    stamp_struct* all_stamps = (stamp_struct*)calloc(state.nS, sizeof(stamp_struct));
    if (allocateStamps(&state, all_stamps, state.nS)) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamps"); return NULL; }
    
    state.kernel_vec = (double**)calloc(state.nCompKer, sizeof(double*));
    for (int i=0; i<state.nCompKer; ++i) state.kernel_vec[i] = (double*)calloc(state.fwKernel*state.fwKernel, sizeof(double));
    getKernelVec(&state);

    for(int i = 0; i < state.nS; i++) {
        PyObject* stamp_tuple = PyList_GetItem(stamps_list, i);
        all_stamps[i].xss[0] = (int)PyFloat_AsDouble(PyTuple_GetItem(stamp_tuple, 0));
        all_stamps[i].yss[0] = (int)PyFloat_AsDouble(PyTuple_GetItem(stamp_tuple, 1));
        all_stamps[i].nss = 1; all_stamps[i].sscnt = 0;
        fillStamp(&state, &all_stamps[i], (float*)PyArray_DATA(conv_arr), (float*)PyArray_DATA(ref_arr));
    }

    double fom = check_stamps(&state, all_stamps, state.nS, (float*)PyArray_DATA(ref_arr), (float*)PyArray_DATA(noise_arr));

    PyObject* result_list = PyList_New(0);
    for(int i = 0; i < state.nS; i++) {
        if (all_stamps[i].diff < state.kerSigReject) {
            PyObject* fit_dict = PyDict_New();
            PyDict_SetItemString(fit_dict, "x", PyFloat_FromDouble(all_stamps[i].xss[all_stamps[i].sscnt]));
            PyDict_SetItemString(fit_dict, "y", PyFloat_FromDouble(all_stamps[i].yss[all_stamps[i].sscnt]));
            PyDict_SetItemString(fit_dict, "figure_of_merit", PyFloat_FromDouble(all_stamps[i].chi2));
            PyList_Append(result_list, fit_dict);
        }
    }
    
    freeStampMem(&state, all_stamps, state.nS); free(all_stamps);
    free_hotpants_state(&state);
    return Py_BuildValue("Od", result_list, fom);
}

static PyObject *py_check_and_refit_stamps(PyObject *self, PyObject *args) {
    PyObject *stamps_list, *config_dict_obj;
    PyArrayObject *conv_arr, *ref_arr, *noise_arr;
    char* conv_dir;

    if (!PyArg_ParseTuple(args, "O!O!O!O!sO!", &PyList_Type, &stamps_list, &PyArray_Type, &conv_arr, &PyArray_Type, &ref_arr, &PyArray_Type, &noise_arr, &conv_dir, &PyDict_Type, &config_dict_obj)) {
        return NULL;
    }
    int ny = (int)PyArray_DIM(conv_arr, 0); int nx = (int)PyArray_DIM(conv_arr, 1);
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);

    state.mRData = (int*)calloc(nx*ny, sizeof(int));
    state.kernel_vec = (double**)calloc(state.nCompKer, sizeof(double*));
    for (int i=0; i<state.nCompKer; ++i) state.kernel_vec[i] = (double*)calloc(state.fwKernel*state.fwKernel, sizeof(double));
    getKernelVec(&state);

    state.nS = PyList_Size(stamps_list);
    stamp_struct* stamps = (stamp_struct*)calloc(state.nS, sizeof(stamp_struct));
    if (allocateStamps(&state, stamps, state.nS)) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamps"); return NULL; }

    for(int i = 0; i < state.nS; ++i) {
        PyObject* stamp_dict = PyList_GetItem(stamps_list, i);
        stamps[i].xss[0] = (int)PyFloat_AsDouble(PyDict_GetItemString(stamp_dict, "x"));
        stamps[i].yss[0] = (int)PyFloat_AsDouble(PyDict_GetItemString(stamp_dict, "y"));
        stamps[i].nss = 1; stamps[i].sscnt = 0;
        fillStamp(&state, &stamps[i], (float*)PyArray_DATA(conv_arr), (float*)PyArray_DATA(ref_arr));
    }
    
    double *kernel_sol = (double*)calloc(state.nCompTotal+1, sizeof(double));
    state.temp = (float*)calloc(state.fwKSStamp*state.fwKSStamp, sizeof(float));
    state.indx = (int*)calloc(state.nCompTotal+1, sizeof(int));
    state.check_mat = (double**)calloc(state.nC+1, sizeof(double*)); for (int i=0; i<=state.nC; i++) state.check_mat[i] = (double*)calloc(state.nC+1, sizeof(double));
    state.check_vec = (double*)calloc(state.nC+1, sizeof(double));
    state.kernel_coeffs = (double*)calloc(state.nCompKer, sizeof(double));
    state.kernel = (double*)calloc(state.fwKernel*state.fwKernel, sizeof(double));

    double meansig_substamps, scatter_substamps;
    int n_skipped_substamps;
    char check_needed = check_again(&state, stamps, kernel_sol, (float*)PyArray_DATA(conv_arr), (float*)PyArray_DATA(ref_arr), (float*)PyArray_DATA(noise_arr), &meansig_substamps, &scatter_substamps, &n_skipped_substamps);
    
    PyObject* new_stamps_list = PyList_New(0);
    for(int i = 0; i < state.nS; i++) {
        if(stamps[i].sscnt < stamps[i].nss) {
            PyObject* stamp_dict = PyDict_New();
            PyDict_SetItemString(stamp_dict, "x", PyFloat_FromDouble(stamps[i].xss[stamps[i].sscnt]));
            PyDict_SetItemString(stamp_dict, "y", PyFloat_FromDouble(stamps[i].yss[stamps[i].sscnt]));
            PyDict_SetItemString(stamp_dict, "figure_of_merit", PyFloat_FromDouble(stamps[i].chi2));
            PyList_Append(new_stamps_list, stamp_dict);
        }
    }
    
    freeStampMem(&state, stamps, state.nS); free(stamps); free(kernel_sol);
    free_hotpants_state(&state);
    return Py_BuildValue("Oii", new_stamps_list, n_skipped_substamps, check_needed);
}

static PyObject *py_get_global_solution(PyObject *self, PyObject *args) {
    PyObject *best_fits_list;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &best_fits_list, &PyDict_Type, &config_dict_obj)) { return NULL; }
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, 1, 1, &config);
    
    int n_stamps = PyList_Size(best_fits_list);
    stamp_struct* stamps = (stamp_struct*)calloc(n_stamps, sizeof(stamp_struct));
    if (allocateStamps(&state, stamps, n_stamps)) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamps"); return NULL; }

    double *kernel_sol = (double*)calloc(state.nCompTotal+1, sizeof(double));
    if (!kernel_sol) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel solution"); return NULL; }

    double** matrix = (double**)calloc(state.nCompTotal + 2, sizeof(double*));
    for (int i=0; i<=state.nCompTotal + 1; ++i) matrix[i] = (double*)calloc(state.nCompTotal + 2, sizeof(double));
    state.wxy = (double**)calloc(n_stamps, sizeof(double*));
    for (int i=0; i<n_stamps; ++i) state.wxy[i] = (double*)calloc(state.nComp, sizeof(double));
    state.indx = (int*)calloc(state.nCompTotal+2, sizeof(int));
    
    // NOTE: The Python code must ensure that stamps are filled before this call.
    build_matrix(&state, stamps, n_stamps, matrix);
    build_scprod(&state, stamps, n_stamps, (float*)NULL, kernel_sol);
    
    double d;
    ludcmp(matrix, state.nCompTotal+1, state.indx, &d);
    lubksb(matrix, state.nCompTotal+1, state.indx, kernel_sol);

    npy_intp dims[] = {state.nCompTotal + 1};
    PyObject* global_coeffs_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, kernel_sol); Py_INCREF(global_coeffs_array);
    
    freeStampMem(&state, stamps, n_stamps); free(stamps); free(kernel_sol);
    for(int i=0; i<=state.nCompTotal+1; ++i) free(matrix[i]); free(matrix);
    for(int i=0; i<n_stamps; ++i) free(state.wxy[i]); free(state.wxy);
    free(state.indx);
    
    free_hotpants_state(&state);
    return global_coeffs_array;
}

static PyObject *py_apply_kernel(PyObject *self, PyObject *args) {
    PyArrayObject *image_arr, *kernel_solution_arr;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &image_arr, &PyArray_Type, &kernel_solution_arr, &PyDict_Type, &config_dict_obj)) { return NULL; }
    int ny = (int)PyArray_DIM(image_arr, 0); int nx = (int)PyArray_DIM(image_arr, 1);
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);
    
    state.mRData = (int*)calloc(nx*ny, sizeof(int)); if (!state.mRData) { free_hotpants_state(&state); PyErr_SetString(PyExc_MemoryError, "Failed to allocate mask"); return NULL; }
    state.kernel_vec = (double**)calloc(state.nCompKer, sizeof(double*));
    for (int i=0; i<state.nCompKer; ++i) state.kernel_vec[i] = (double*)calloc(state.fwKernel*state.fwKernel, sizeof(double));
    getKernelVec(&state);
    state.kernel = (double*)calloc(state.fwKernel*state.fwKernel, sizeof(double)); state.kernel_coeffs = (double*)calloc(state.nCompKer, sizeof(double));
    double* kernel_sol = (double*)PyArray_DATA(kernel_solution_arr);
    float* conv_image = (float*)calloc(nx * ny, sizeof(float)); float* variance_image = (float*)calloc(nx*ny, sizeof(float)); 
    
    spatial_convolve(&state, (float*)PyArray_DATA(image_arr), &variance_image, nx, ny, kernel_sol, conv_image, state.mRData);
    
    npy_intp dims[] = {ny, nx};
    PyObject* conv_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, conv_image); Py_INCREF(conv_arr);
    PyObject* mask_arr = PyArray_SimpleNewFromData(2, dims, NPY_INT32, state.mRData); Py_INCREF(mask_arr);

    for(int i=0; i<state.nCompKer; ++i) free(state.kernel_vec[i]); free(state.kernel_vec);
    free(state.kernel); free(state.kernel_coeffs); free_hotpants_state(&state);
    
    return Py_BuildValue("OO", conv_arr, mask_arr);
}

static PyObject *py_get_background_image(PyObject *self, PyObject *args) {
    PyArrayObject *shape_arr, *kernel_sol_arr;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &shape_arr, &PyArray_Type, &kernel_sol_arr, &PyDict_Type, &config_dict_obj)) { return NULL; }
    int ny = (int)PyArray_DIM(shape_arr, 0); int nx = (int)PyArray_DIM(shape_arr, 1);
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);
    
    double *kernel_sol = (double*)PyArray_DATA(kernel_sol_arr);
    float* bkg_image = (float*)calloc(nx*ny, sizeof(float));

    for(int j=0; j<ny; ++j) { for(int i=0; i<nx; ++i) { bkg_image[i + nx*j] = get_background(&state, i, j, kernel_sol); } }
    npy_intp dims[] = {ny, nx};
    PyObject* bkg_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, bkg_image); Py_INCREF(bkg_arr);

    free_hotpants_state(&state);
    return bkg_arr;
}

static PyObject *py_rescale_noise_ok(PyObject *self, PyObject *args) {
    PyArrayObject *diff_arr, *noise_arr, *mask_arr;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &diff_arr, &PyArray_Type, &noise_arr, &PyArray_Type, &mask_arr, &PyDict_Type, &config_dict_obj)) { return NULL; }
    int ny = (int)PyArray_DIM(diff_arr, 0); int nx = (int)PyArray_DIM(diff_arr, 1);
    hotpants_config_t config; 
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);
    state.mRData = (int*)PyArray_DATA(mask_arr);

    float *diff_data = (float*)PyArray_DATA(diff_arr); float *noise_data = (float*)PyArray_DATA(noise_arr); int *mask_data = (int*)PyArray_DATA(mask_arr);
    double sdm, nmeanm;
    getStampStats3(&state, diff_data, 0, 0, nx, ny, NULL, &sdm, NULL, NULL, NULL, NULL, NULL, 0xff, FLAG_OUTPUT_ISBAD, 5);
    getStampStats3(&state, noise_data, 0, 0, nx, ny, NULL, &nmeanm, NULL, NULL, NULL, NULL, NULL, 0xff, FLAG_OUTPUT_ISBAD, 5);
    double diff_rat = sdm / nmeanm;
    if (diff_rat > 1) { for(int i=0; i<nx*ny; i++) { if((mask_data[i] & 0xff) && (!(mask_data[i] & FLAG_OUTPUT_ISBAD))) { noise_data[i] *= diff_rat; } } }
    npy_intp dims[] = {ny, nx};
    PyObject* new_noise_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, noise_data); Py_INCREF(new_noise_arr);
    
    free_hotpants_state(&state);
    return new_noise_arr;
}

static PyObject *py_calculate_final_stats(PyObject *self, PyObject *args) {
    PyArrayObject *diff_arr, *noise_arr, *mask_arr;
    PyObject *config_dict_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &diff_arr, &PyArray_Type, &noise_arr, &PyArray_Type, &mask_arr, &PyDict_Type, &config_dict_obj)) { return NULL; }
    int ny = (int)PyArray_DIM(diff_arr, 0); int nx = (int)PyArray_DIM(diff_arr, 1);
    hotpants_config_t config;
    if(parse_config_dict(config_dict_obj, &config) < 0) return NULL;
    hotpants_state_t state; init_hotpants_state(&state, nx, ny, &config);
    state.mRData = (int*)PyArray_DATA(mask_arr);
    
    float *diff_data = (float*)PyArray_DATA(diff_arr); float *noise_data = (float*)PyArray_DATA(noise_arr);
    double sum, mean, median, mode, sd, fwhm, lfwhm;
    double nsum, nmean, nmedian, nmode, nsd, nfwhm, nlfwhm;
    double x2norm; int nx2norm;
    
    getStampStats3(&state, diff_data, 0, 0, nx, ny, &sum, &mean, &median, &mode, &sd, &fwhm, &lfwhm, 0x0, 0xffff, 5);
    getStampStats3(&state, noise_data, 0, 0, nx, ny, &nsum, &nmean, &nmedian, &nmode, &nsd, &nfwhm, &nlfwhm, 0x0, 0xffff, 5);
    getNoiseStats3(&state, diff_data, noise_data, &x2norm, &nx2norm, 0x0, 0xffff);
    
    PyObject* stats_dict = PyDict_New();
    PyDict_SetItemString(stats_dict, "diff_mean", PyFloat_FromDouble(mean)); PyDict_SetItemString(stats_dict, "diff_std", PyFloat_FromDouble(sd));
    PyDict_SetItemString(stats_dict, "noise_mean", PyFloat_FromDouble(nmean)); PyDict_SetItemString(stats_dict, "x2norm", PyFloat_FromDouble(x2norm));
    PyDict_SetItemString(stats_dict, "nx2norm", PyLong_FromLong(nx2norm));
    free_hotpants_state(&state);
    return stats_dict;
}

// Helper function definitions
static int parse_config_dict(PyObject *config_dict, hotpants_config_t *config) {
    if (!PyDict_Check(config_dict)) { PyErr_SetString(PyExc_TypeError, "Configuration must be a dictionary"); return -1; }
    config->tuthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tuthresh"));
    PyObject *tuktresh_obj = PyDict_GetItemString(config_dict, "tuktresh");
    config->tuktresh = (tuktresh_obj == Py_None) ? config->tuthresh : PyFloat_AsDouble(tuktresh_obj);
    config->tlthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tlthresh")); config->tgain = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tgain"));
    config->trdnoise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "trdnoise")); config->tpedestal = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tpedestal"));
    config->iuthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "iuthresh"));
    PyObject *iuktresh_obj = PyDict_GetItemString(config_dict, "iuktresh");
    config->iuktresh = (iuktresh_obj == Py_None) ? config->iuthresh : PyFloat_AsDouble(iuktresh_obj);
    config->ilthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ilthresh")); config->igain = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "igain"));
    config->irdnoise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "irdnoise")); config->ipedestal = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ipedestal"));
    config->rkernel = PyLong_AsLong(PyDict_GetItemString(config_dict, "rkernel")); config->ko = PyLong_AsLong(PyDict_GetItemString(config_dict, "ko"));
    config->bgo = PyLong_AsLong(PyDict_GetItemString(config_dict, "bgo")); config->fitthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fitthresh"));
    config->scale_fitthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "scale_fitthresh"));
    config->min_frac_stamps = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "min_frac_stamps"));
    config->nss = PyLong_AsLong(PyDict_GetItemString(config_dict, "nss")); config->rss = PyLong_AsLong(PyDict_GetItemString(config_dict, "rss"));
    config->ks = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ks")); config->kfm = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "kfm"));
    config->verbose = PyLong_AsLong(PyDict_GetItemString(config_dict, "verbose"));
    const char *force_conv_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "force_convolve")); strncpy(config->force_convolve, force_conv_str, 1); config->force_convolve[1] = '\0';
    const char *normalize_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "normalize")); strncpy(config->normalize, normalize_str, 1); config->normalize[1] = '\0';
    const char *fom_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "fom")); strncpy(config->fom, fom_str, 1); config->fom[1] = '\0';
    config->fillval = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fillval")); config->fillval_noise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fillval_noise"));
    config->rescale_ok = PyLong_AsLong(PyDict_GetItemString(config_dict, "rescale_ok"));
    config->conv_var = PyLong_AsLong(PyDict_GetItemString(config_dict, "conv_var"));
    config->use_pca = PyLong_AsLong(PyDict_GetItemString(config_dict, "use_pca"));
    config->nregx = PyLong_AsLong(PyDict_GetItemString(config_dict, "nregx")); config->nregy = PyLong_AsLong(PyDict_GetItemString(config_dict, "nregy"));
    config->nstampx = PyLong_AsLong(PyDict_GetItemString(config_dict, "nstampx")); config->nstampy = PyLong_AsLong(PyDict_GetItemString(config_dict, "nstampy"));
    config->ngauss = PyLong_AsLong(PyDict_GetItemString(config_dict, "ngauss"));
    PyObject* deg_list = PyDict_GetItemString(config_dict, "deg_fixe");
    if(deg_list) { config->deg_fixe = (int*)malloc(config->ngauss * sizeof(int)); for(int i=0; i<config->ngauss; ++i) config->deg_fixe[i] = PyLong_AsLong(PyList_GetItem(deg_list, i)); }
    PyObject* sig_list = PyDict_GetItemString(config_dict, "sigma_gauss");
    if(sig_list) { config->sigma_gauss = (float*)malloc(config->ngauss * sizeof(float)); for(int i=0; i<config->ngauss; ++i) config->sigma_gauss[i] = PyFloat_AsDouble(PyList_GetItem(sig_list, i)); }
    config->fwkernel = PyLong_AsLong(PyDict_GetItemString(config_dict, "fwkernel")); config->fwksstamp = PyLong_AsLong(PyDict_GetItemString(config_dict, "fwksstamp"));
    config->ncomp_ker = PyLong_AsLong(PyDict_GetItemString(config_dict, "ncomp_ker")); config->ncomp = PyLong_AsLong(PyDict_GetItemString(config_dict, "ncomp"));
    config->n_bg_vectors = PyLong_AsLong(PyDict_GetItemString(config_dict, "n_bg_vectors"));
    config->n_comp_total = PyLong_AsLong(PyDict_GetItemString(config_dict, "n_comp_total"));
    config->stat_sig = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "stat_sig"));
    config->kf_spread_mask1 = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "kf_spread_mask1"));
    return 0;
}
static void init_hotpants_state(hotpants_state_t *state, int nx, int ny, const hotpants_config_t *config) {
    memset(state, 0, sizeof(hotpants_state_t)); state->nx = nx; state->ny = ny; state->rPixX = nx; state->rPixY = ny;
    state->tUThresh = config->tuthresh; state->tLThresh = config->tlthresh; state->tUKThresh = config->tuktresh;
    state->tGain = config->tgain; state->tRdnoise = config->trdnoise; state->tPedestal = config->tpedestal;
    state->iUThresh = config->iuthresh; state->iLThresh = config->ilthresh; state->iUKThresh = config->iuktresh;
    state->iGain = config->igain; state->iRdnoise = config->irdnoise; state->iPedestal = config->ipedestal;
    state->hwKernel = config->rkernel; state->kerOrder = config->ko; state->bgOrder = config->bgo;
    state->kerFitThresh = config->fitthresh; state->kerSigReject = config->ks; state->kerFracMask = config->kfm;
    state->nKSStamps = config->nss; state->hwKSStamp = config->rss; state->nRegX = config->nregx; state->nRegY = config->nregy;
    state->nStampX = config->nstampx; state->nStampY = config->nstampy;
    strncpy(state->forceConvolve_str, config->force_convolve, 2);
    strncpy(state->photNormalize_str, config->normalize, 2);
    strncpy(state->figMerit_str, config->fom, 2);
    state->fillVal = config->fillval; state->fillValNoise = config->fillval_noise; state->verbose = config->verbose;
    state->rescaleOK = config->rescale_ok; state->convolveVariance = config->conv_var; state->usePCA = config->use_pca;
    state->statSig = config->stat_sig; state->kfSpreadMask1 = config->kf_spread_mask1;
    state->ngauss = config->ngauss;
    state->deg_fixe = (int*)malloc(state->ngauss * sizeof(int)); memcpy(state->deg_fixe, config->deg_fixe, state->ngauss * sizeof(int));
    state->sigma_gauss = (float*)malloc(state->ngauss * sizeof(float)); memcpy(state->sigma_gauss, config->sigma_gauss, state->ngauss * sizeof(float));
    state->fwKernel = config->fwkernel; 
    
    // The following is a derived value in the C code
    state->fwStamp = 2 * state->hwKernel + 1;
    
    state->hwStamp = state->hwKernel;
    state->fwKSStamp = config->fwksstamp; state->nStamps = state->nStampX * state->nStampY; state->nCompKer = config->ncomp_ker;
    state->nComp = config->ncomp; state->nBGVectors = config->n_bg_vectors; state->nCompTotal = config->n_comp_total;
    state->nC = state->nCompKer + state->nBGVectors + 1;
    state->mRData = (int*)calloc(nx * ny, sizeof(int)); state->temp = (float*)calloc(nx*ny, sizeof(float)); state->temp2 = (float*)calloc(nx*ny, sizeof(float));
    state->indx = (int*)calloc(state->nCompTotal + 100, sizeof(int)); state->kernel = (double*)calloc(state->fwKernel*state->fwKernel, sizeof(double));
    state->kernel_coeffs = (double*)calloc(state->nCompKer, sizeof(double));
    state->kernel_vec = (double**)calloc(state->nCompKer, sizeof(double*));
    state->filter_x = (double*)calloc(state->nCompKer*state->fwKernel, sizeof(double)); state->filter_y = (double*)calloc(state->nCompKer*state->fwKernel, sizeof(double));
    state->check_mat = (double**)calloc(state->nC, sizeof(double*)); for (int i=0; i<state->nC; ++i) state->check_mat[i] = (double*)calloc(state->nC, sizeof(double));
    state->check_vec = (double*)calloc(state->nC, sizeof(double)); state->check_stack = (double*)calloc(state->nStamps, sizeof(double));
    state->wxy = (double**)calloc(state->nStamps, sizeof(double*)); for (int i=0; i<state->nStamps; ++i) state->wxy[i] = (double*)calloc(state->nComp, sizeof(double));
    state->PCA = (float**)calloc(state->ngauss, sizeof(float*));

    // Allocate the stamp arrays which are global in the original C code
    state->tStamps = (stamp_struct*)calloc(state->nStamps, sizeof(stamp_struct));
    allocateStamps(state, state->tStamps, state->nStamps);
    state->iStamps = (stamp_struct*)calloc(state->nStamps, sizeof(stamp_struct));
    allocateStamps(state, state->iStamps, state->nStamps);
}
static void free_hotpants_state(hotpants_state_t *state) {
    if (state->mRData) free(state->mRData); if (state->temp) free(state->temp); if (state->temp2) free(state->temp2);
    if (state->indx) free(state->indx); if (state->kernel) free(state->kernel); if (state->kernel_coeffs) free(state->kernel_coeffs);
    if (state->kernel_vec) { for(int i=0; i<state->nCompKer; ++i) if(state->kernel_vec[i]) free(state->kernel_vec[i]); free(state->kernel_vec); }
    if (state->filter_x) free(state->filter_x); if (state->filter_y) free(state->filter_y);
    if (state->check_mat) { for(int i=0; i<state->nC; ++i) if(state->check_mat[i]) free(state->check_mat[i]); free(state->check_mat); }
    if (state->check_vec) free(state->check_vec); if (state->check_stack) free(state->check_stack);
    if (state->wxy) { for(int i=0; i<state->nStamps; ++i) if(state->wxy[i]) free(state->wxy[i]); free(state->wxy); }
    if (state->PCA) { for(int i=0; i<state->ngauss; ++i) if(state->PCA[i]) free(state->PCA[i]); free(state->PCA); }
    if (state->tStamps) { freeStampMem(&state, state->tStamps, state->nStamps); free(state->tStamps); }
    if (state->iStamps) { freeStampMem(&state, state->iStamps, state->nStamps); free(state->iStamps); }
    if (state->deg_fixe) free(state->deg_fixe); if(state->sigma_gauss) free(state->sigma_gauss);
}
static PyMethodDef hotpants_ext_methods[] = {
    {"make_input_mask", py_make_input_mask, METH_VARARGS, "Creates a pixel mask based on input data and thresholds."},
    {"calculate_noise_image", py_calculate_noise_image, METH_VARARGS, "Calculates a noise image based on gain and readnoise."},
    {"find_stamps", py_find_stamps, METH_VARARGS, "Finds potential stamps for kernel fitting."},
    {"fit_stamps_and_get_fom", py_fit_stamps_and_get_fom, METH_VARARGS, "Fits individual stamps and returns figures of merit."},
    {"check_and_refit_stamps", py_check_and_refit_stamps, METH_VARARGS, "Performs iterative check and refit of stamps."},
    {"get_global_solution", py_get_global_solution, METH_VARARGS, "Computes global kernel solution."},
    {"apply_kernel", py_apply_kernel, METH_VARARGS, "Applies a global kernel solution to convolve an image."},
    {"get_background_image", py_get_background_image, METH_VARARGS, "Calculates the background image from the kernel solution."},
    {"rescale_noise_ok", py_rescale_noise_ok, METH_VARARGS, "Rescales noise for 'OK' pixels to match empirical noise ratio."},
    {"calculate_final_stats", py_calculate_final_stats, METH_VARARGS, "Calculates final statistics on the difference image."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef hotpants_ext_module = {
    PyModuleDef_HEAD_INIT, "hotpants_ext", "HOTPanTS C extension for image differencing", -1, hotpants_ext_methods
};
PyMODINIT_FUNC PyInit_hotpants_ext(void) {
    PyObject *module = PyModule_Create(&hotpants_ext_module); if (module == NULL) return NULL; import_array(); return module;
}
