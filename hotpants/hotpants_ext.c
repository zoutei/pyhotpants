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

/*****************************************************
 * New state management functions and helper structures
 *****************************************************/
// Structure to hold hotpants configuration from Python
typedef struct
{
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

// Structure to hold hotpants_state_t with a reference count for Python
typedef struct
{
    PyObject_HEAD hotpants_state_t *state;
} HotpantsStateObject;

static PyObject *hotpants_state_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void hotpants_state_dealloc(HotpantsStateObject *self);
static PyObject *hotpants_state_init_from_config(PyObject *self, PyObject *args);

static PyMethodDef hotpants_state_methods[] = {
    {NULL} /* Sentinel */
};

static PyTypeObject HotpantsState_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "hotpants_ext.HotpantsState", // Add this line
    .tp_basicsize = sizeof(HotpantsStateObject),
    .tp_dealloc = (destructor)hotpants_state_dealloc,
    .tp_new = hotpants_state_new,
    .tp_init = (initproc)hotpants_state_init_from_config,
    .tp_methods = hotpants_state_methods,
};

// Function prototypes
static int parse_config_dict(PyObject *config_dict, hotpants_config_t *config);
static void init_hotpants_state(hotpants_state_t *state, int nx, int ny, const hotpants_config_t *config);
static void free_hotpants_state(HotpantsStateObject *self);
void allocateStampMem(hotpants_state_t *state, stamp_struct *stamps, int n);
void freeStampMem(hotpants_state_t *state, stamp_struct *stamps, int n);

static PyObject *hotpants_state_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    HotpantsStateObject *self;
    self = (HotpantsStateObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->state = (hotpants_state_t *)malloc(sizeof(hotpants_state_t));
        if (self->state == NULL)
        {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate hotpants_state_t");
            return NULL;
        }
        memset(self->state, 0, sizeof(hotpants_state_t));
    }
    return (PyObject *)self;
}

static void hotpants_state_dealloc(HotpantsStateObject *self)
{
    if (self->state)
    {
        // Free all memory owned by the C state object
        if (self->state->mRData)
            free(self->state->mRData);
        if (self->state->temp)
            free(self->state->temp);
        if (self->state->temp2)
            free(self->state->temp2);
        if (self->state->indx)
            free(self->state->indx);
        if (self->state->kernel)
            free(self->state->kernel);
        if (self->state->kernel_coeffs)
            free(self->state->kernel_coeffs);
        if (self->state->kernel_vec)
        {
            for (int i = 0; i < self->state->nCompKer; ++i)
                if (self->state->kernel_vec[i])
                    free(self->state->kernel_vec[i]);
            free(self->state->kernel_vec);
        }
        if (self->state->filter_x)
            free(self->state->filter_x);
        if (self->state->filter_y)
            free(self->state->filter_y);
        if (self->state->check_mat)
        {
            for (int i = 0; i < self->state->nC; ++i)
                if (self->state->check_mat[i])
                    free(self->state->check_mat[i]);
            free(self->state->check_mat);
        }
        if (self->state->check_vec)
            free(self->state->check_vec);
        if (self->state->check_stack)
            free(self->state->check_stack);
        if (self->state->wxy)
        {
            for (int i = 0; i < self->state->nStamps; ++i)
                if (self->state->wxy[i])
                    free(self->state->wxy[i]);
            free(self->state->wxy);
        }
        if (self->state->PCA)
        {
            for (int i = 0; i < self->state->ngauss; ++i)
                if (self->state->PCA[i])
                    free(self->state->PCA[i]);
            free(self->state->PCA);
        }
        if (self->state->tStamps)
        {
            freeStampMem(self->state, self->state->tStamps, self->state->nStamps);
            free(self->state->tStamps);
        }
        if (self->state->iStamps)
        {
            freeStampMem(self->state, self->state->iStamps, self->state->nStamps);
            free(self->state->iStamps);
        }
        if (self->state->deg_fixe)
            free(self->state->deg_fixe);
        if (self->state->sigma_gauss)
            free(self->state->sigma_gauss);

        free(self->state);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// Function to initialize the state object from a Python dict
static PyObject *hotpants_state_init_from_config(PyObject *self, PyObject *args)
{
    PyObject *config_dict_obj;
    int nx, ny;
    if (!PyArg_ParseTuple(args, "iiO!", &nx, &ny, &PyDict_Type, &config_dict_obj))
    {
        return NULL;
    }

    HotpantsStateObject *state_obj = (HotpantsStateObject *)self;
    if (!state_obj || !state_obj->state)
    {
        PyErr_SetString(PyExc_RuntimeError, "Invalid state object.");
        return NULL;
    }

    hotpants_config_t config;
    if (parse_config_dict(config_dict_obj, &config) < 0)
    {
        return NULL;
    }

    // Allocate and initialize the C state
    init_hotpants_state(state_obj->state, nx, ny, &config);
    Py_RETURN_NONE;
}

/*****************************************************
 * Python function wrappers
 *****************************************************/
static PyObject *py_make_input_mask(PyObject *self, PyObject *args)
{
    PyArrayObject *template_arr, *image_arr;
    PyObject *t_mask_obj, *i_mask_obj;
    HotpantsStateObject *state_obj;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
                          &HotpantsState_Type, &state_obj,
                          &PyArray_Type, &template_arr,
                          &PyArray_Type, &image_arr,
                          &PyArray_Type, &t_mask_obj,
                          &PyArray_Type, &i_mask_obj))
    {
        return NULL;
    }

    hotpants_state_t *state = state_obj->state;

    // Allocate the mask array that will be returned to Python
    int *mData = (int *)calloc(state->nx * state->ny, sizeof(int));
    if (!mData)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate mData array");
        return NULL;
    }

    // First, incorporate external masks if they exist, replicating main.c behavior
    if (t_mask_obj && t_mask_obj != Py_None)
    {
        int *t_mask_data = (int *)PyArray_DATA((PyArrayObject *)t_mask_obj);
        for (int i = 0; i < state->nx * state->ny; ++i)
        {
            if (t_mask_data[i] > 0)
            {
                mData[i] |= FLAG_INPUT_MASK;
            }
        }
    }
    if (i_mask_obj && i_mask_obj != Py_None)
    {
        int *i_mask_data = (int *)PyArray_DATA((PyArrayObject *)i_mask_obj);
        for (int i = 0; i < state->nx * state->ny; ++i)
        {
            if (i_mask_data[i] > 0)
            {
                mData[i] |= FLAG_INPUT_MASK;
            }
        }
    }

    // Then, call the core C function to generate the mask based on image data and thresholds.
    makeInputMask(state, (float *)PyArray_DATA(template_arr), (float *)PyArray_DATA(image_arr), mData);

    // Finally, mask the borders, as per the original main.c
    int sBorder = state->hwKSStamp + state->hwKernel;
    for (int l = 0; l < state->ny; l++)
    {
        for (int k = 0; k < sBorder; k++)
        {
            mData[k + state->nx * l] |= (FLAG_T_BAD | FLAG_I_BAD);
        }
        for (int k = state->nx - sBorder; k < state->nx; k++)
        {
            mData[k + state->nx * l] |= (FLAG_T_BAD | FLAG_I_BAD);
        }
    }
    for (int l = 0; l < sBorder; l++)
    {
        for (int k = sBorder; k < state->nx - sBorder; k++)
        {
            mData[k + state->nx * l] |= (FLAG_T_BAD | FLAG_I_BAD);
        }
    }
    for (int l = state->ny - sBorder; l < state->ny; l++)
    {
        for (int k = sBorder; k < state->nx - sBorder; k++)
        {
            mData[k + state->nx * l] |= (FLAG_T_BAD | FLAG_I_BAD);
        }
    }

    npy_intp dims[2] = {state->ny, state->nx};
    PyObject *mask_arr = PyArray_SimpleNewFromData(2, dims, NPY_INT32, mData);
    if (!mask_arr)
    {
        free(mData);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array from data");
        return NULL;
    }
    Py_INCREF(mask_arr);
    PyArray_ENABLEFLAGS((PyArrayObject *)mask_arr, NPY_ARRAY_OWNDATA);

    return mask_arr;
}

static PyObject *py_calculate_noise_image(PyObject *self, PyObject *args)
{
    PyArrayObject *image_arr;
    int is_template;
    HotpantsStateObject *state_obj;

    if (!PyArg_ParseTuple(args, "O!O!p", &HotpantsState_Type, &state_obj, &PyArray_Type, &image_arr, &is_template))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    float *data = (float *)PyArray_DATA(image_arr);
    float inv_gain = is_template ? 1.0 / state->tGain : 1.0 / state->iGain;
    float rdnoise = is_template ? state->tRdnoise / state->tGain : state->iRdnoise / state->iGain;

    float *noise_data = makeNoiseImage4(state, data, inv_gain, rdnoise);
    if (!noise_data)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate noise data");
        return NULL;
    }

    npy_intp dims[2] = {state->ny, state->nx};
    PyObject *noise_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, noise_data);
    if (!noise_arr)
    {
        free(noise_data);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array from data");
        return NULL;
    }
    Py_INCREF(noise_arr);
    PyArray_ENABLEFLAGS((PyArrayObject *)noise_arr, NPY_ARRAY_OWNDATA);

    return noise_arr;
}

static PyObject *py_find_stamps(PyObject *self, PyObject *args)
{
    PyArrayObject *template_arr, *image_arr, *input_mask_arr;
    PyArrayObject *catalog_arr_obj;
    double fit_thresh;
    HotpantsStateObject *state_obj;

    if (!PyArg_ParseTuple(args, "O!O!O!O!dO",
                          &HotpantsState_Type, &state_obj,
                          &PyArray_Type, &template_arr,
                          &PyArray_Type, &image_arr,
                          &PyArray_Type, &input_mask_arr,
                          &fit_thresh,
                          &PyArray_Type, &catalog_arr_obj))
    {
        return NULL;
    }

    hotpants_state_t *state = state_obj->state;
    state->mRData = (int *)PyArray_DATA(input_mask_arr);
    state->kerFitThresh = fit_thresh;

    int niS = 0, ntS = 0;
    int rXMin = 0, rYMin = 0, rXMax = state->nx - 1, rYMax = state->ny - 1;

    state->tStamps = (stamp_struct *)calloc(state->nStamps, sizeof(stamp_struct));
    if (allocateStamps(state, state->tStamps, state->nStamps))
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamp vector");
        free(state->tStamps);
        state->tStamps = NULL;
        return NULL;
    }

    state->iStamps = (stamp_struct *)calloc(state->nStamps, sizeof(stamp_struct));
    if (allocateStamps(state, state->iStamps, state->nStamps))
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamp vector");
        freeStampMem(state, state->tStamps, state->nStamps);
        free(state->tStamps);
        state->tStamps = NULL;
        free(state->iStamps);
        state->iStamps = NULL;
        return NULL;
    }

    if (catalog_arr_obj && PyArray_Check(catalog_arr_obj))
    {
        // Get the data pointer and dimensions from the NumPy array
        float *catalog_data = (float *)PyArray_DATA((PyArrayObject *)catalog_arr_obj);
        npy_intp *dims = PyArray_DIMS((PyArrayObject *)catalog_arr_obj);
        int num_catalog_entries = dims[0];

        for (int i = 0; i < num_catalog_entries; ++i)
        {
            double x_pos = catalog_data[i * 2];     // Access x coordinate
            double y_pos = catalog_data[i * 2 + 1]; // Access y coordinate
            // Replicating main.c logic: use 0 for no automatic search
            buildStamps(state, rXMin, rXMax, rYMin, rYMax, &niS, &ntS, 0, rXMin, rYMin,
                        state->iStamps, state->tStamps, (float *)PyArray_DATA(image_arr),
                        (float *)PyArray_DATA(template_arr), (int)x_pos, (int)y_pos);

            if (!(strncmp(state->forceConvolve_str, "i", 1) == 0))
            {
                if (state->tStamps[ntS].nss > 0)
                    ntS += 1;
            }
            if (!(strncmp(state->forceConvolve_str, "t", 1) == 0))
            {
                if (state->iStamps[niS].nss > 0)
                    niS += 1;
            }
        }
        printf("Found %d catalog entries.\n", num_catalog_entries);
    }
    else
    {
        // Original logic for automated grid search
        for (int l = 0; l < state->nStampY; l++)
        {
            for (int k = 0; k < state->nStampX; k++)
            {
                int sXMin = rXMin + (int)(k * (rXMax - rXMin + 1.0) / state->nStampX);
                int sYMin = rYMin + (int)(l * (rYMax - rYMin + 1.0) / state->nStampY);
                int sXMax = imin(sXMin + state->fwStamp - 1, rXMax);
                int sYMax = imin(sYMin + state->fwStamp - 1, rYMax);

                if (!(strncmp(state->forceConvolve_str, "i", 1) == 0))
                {
                    state->tStamps[ntS].sscnt = state->tStamps[ntS].nss = 0;
                    state->tStamps[ntS].chi2 = 0.0;
                }
                if (!(strncmp(state->forceConvolve_str, "t", 1) == 0))
                {
                    state->iStamps[niS].sscnt = state->iStamps[niS].nss = 0;
                    state->iStamps[niS].chi2 = 0.0;
                }

                // Replicating main.c logic: use 1 for automatic search
                buildStamps(state, sXMin, sXMax, sYMin, sYMax, &niS, &ntS, 1, rXMin, rYMin,
                            state->iStamps, state->tStamps, (float *)PyArray_DATA(image_arr),
                            (float *)PyArray_DATA(template_arr), 0.0, 0.0);

                if (!(strncmp(state->forceConvolve_str, "i", 1) == 0))
                {
                    if (state->tStamps[ntS].nss > 0)
                        ntS += 1;
                }
                if (!(strncmp(state->forceConvolve_str, "t", 1) == 0))
                {
                    if (state->iStamps[niS].nss > 0)
                        niS += 1;
                }
            }
        }
    }
    printf("Found %d template stamps and %d image stamps.\n", ntS, niS);

    PyObject *t_stamps_list = PyList_New(0);
    if (!(strncmp(state->forceConvolve_str, "i", 1) == 0)) {
        for (int i = 0; i < ntS; ++i) {
            PyObject *stamp_dict = PyDict_New();
            PyObject *substamps = PyList_New(0);
            for (int j = 0; j < state->tStamps[i].nss; j++) {
                PyObject *substamp_tuple = PyTuple_New(2);
                PyTuple_SetItem(substamp_tuple, 0, PyFloat_FromDouble(state->tStamps[i].xss[j]));
                PyTuple_SetItem(substamp_tuple, 1, PyFloat_FromDouble(state->tStamps[i].yss[j]));
                PyList_Append(substamps, substamp_tuple);
            }
            PyDict_SetItemString(stamp_dict, "substamps", substamps);
            PyList_Append(t_stamps_list, stamp_dict);
        }
    }

    PyObject *i_stamps_list = PyList_New(0);
    if (!(strncmp(state->forceConvolve_str, "t", 1) == 0)) {
        for (int i = 0; i < niS; ++i) {
            PyObject *stamp_dict = PyDict_New();
            PyObject *substamps = PyList_New(0);
            for (int j = 0; j < state->iStamps[i].nss; j++) {
                PyObject *substamp_tuple = PyTuple_New(2);
                PyTuple_SetItem(substamp_tuple, 0, PyFloat_FromDouble(state->iStamps[i].xss[j]));
                PyTuple_SetItem(substamp_tuple, 1, PyFloat_FromDouble(state->iStamps[i].yss[j]));
                PyList_Append(substamps, substamp_tuple);
            }
            PyDict_SetItemString(stamp_dict, "substamps", substamps);
            PyList_Append(i_stamps_list, stamp_dict);
        }
    }

    return Py_BuildValue("OO", t_stamps_list, i_stamps_list);
}

static PyObject *py_fit_stamps_and_get_fom(PyObject *self, PyObject *args)
{
    PyArrayObject *conv_arr, *ref_arr, *noise_arr;
    PyObject *stamps_list;
    HotpantsStateObject *state_obj;
    char *conv_dir;

    if (!PyArg_ParseTuple(args, "O!O!O!O!sO!", &HotpantsState_Type, &state_obj, &PyArray_Type, &conv_arr, &PyArray_Type, &ref_arr, &PyArray_Type, &noise_arr, &conv_dir, &PyList_Type, &stamps_list))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    // We assume the state is initialized, but some temporary arrays might be null
    // Here we allocate only what's necessary for this function
    if (!state->kernel_vec)
    {
        state->kernel_vec = (double **)calloc(state->nCompKer, sizeof(double *));
        if (!state->kernel_vec)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel_vec");
            return NULL;
        }
    }
    getKernelVec(state);

    state->nS = PyList_Size(stamps_list);
    stamp_struct *all_stamps = (stamp_struct *)calloc(state->nS, sizeof(stamp_struct));
    if (allocateStamps(state, all_stamps, state->nS))
    {
        free(all_stamps);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamps");
        return NULL;
    }

    for (int i = 0; i < state->nS; i++)
    {
        PyObject *stamp_dict = PyList_GetItem(stamps_list, i);
        PyObject *substamps_list = PyDict_GetItemString(stamp_dict, "substamps");
        int n_substamps = PyList_Size(substamps_list);

        all_stamps[i].nss = n_substamps;
        all_stamps[i].sscnt = 0;

        for (int j = 0; j < n_substamps; j++) {
            PyObject *substamp_tuple = PyList_GetItem(substamps_list, j);
            all_stamps[i].xss[j] = (int)PyFloat_AsDouble(PyTuple_GetItem(substamp_tuple, 0));
            all_stamps[i].yss[j] = (int)PyFloat_AsDouble(PyTuple_GetItem(substamp_tuple, 1));
        }
        
        if (all_stamps[i].nss > 0) {
            fillStamp(state, &all_stamps[i], (float *)PyArray_DATA(conv_arr), (float *)PyArray_DATA(ref_arr));
        }
    }
    state->mRData = (int *)PyArray_DATA(noise_arr); // Dummy use for check_stamps to access mask
    double fom = check_stamps(state, all_stamps, state->nS, (float *)PyArray_DATA(ref_arr), (float *)PyArray_DATA(noise_arr));

    PyObject *result_list = PyList_New(0);
    if (!result_list)
    {
        freeStampMem(state, all_stamps, state->nS);
        free(all_stamps);
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list");
        return NULL;
    }
    for (int i = 0; i < state->nS; i++)
    {
        if (all_stamps[i].diff < state->kerSigReject)
        {
            PyObject *good_stamp_dict = PyList_GetItem(stamps_list, i);
            Py_INCREF(good_stamp_dict);
            PyList_Append(result_list, good_stamp_dict);
        }
    }

    return Py_BuildValue("Od", result_list, fom);
}
/*
 * This function is a wrapper around the core HOTPANTS kernel fitting logic,
 * which is equivalent to the `fitKernel` function in the original C code.
 * It performs iterative sigma-clipping of substamps and then computes the
 * final global kernel solution.
 */
static PyObject *py_fit_kernel(PyObject *self, PyObject *args)
{
    PyObject *stamps_list;
    PyArrayObject *conv_arr, *ref_arr, *noise_arr, *mask_arr;
    HotpantsStateObject *state_obj;

    // Parse Python arguments (no change here)
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                          &HotpantsState_Type, &state_obj,
                          &PyList_Type, &stamps_list,
                          &PyArray_Type, &conv_arr,
                          &PyArray_Type, &ref_arr,
                          &PyArray_Type, &noise_arr,
                          &PyArray_Type, &mask_arr))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    // Set the mask data for the C functions that need it
    state->mRData = (int *)PyArray_DATA(mask_arr);

    // 1. Setup stamps from Python list
    int n_stamps = PyList_Size(stamps_list);
    state->nS = n_stamps; // Set nS in state, as it's used by other functions
    stamp_struct *stamps = (stamp_struct *)calloc(n_stamps, sizeof(stamp_struct));
    if (!stamps || allocateStamps(state, stamps, n_stamps))
    {
        free(stamps);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate stamps");
        return NULL;
    }

    for (int i = 0; i < n_stamps; ++i)
    {
        PyObject *stamp_dict = PyList_GetItem(stamps_list, i);
        PyObject *substamps_list = PyDict_GetItemString(stamp_dict, "substamps");
        int n_substamps = PyList_Size(substamps_list);

        stamps[i].nss = n_substamps;
        stamps[i].sscnt = 0;

        for (int j = 0; j < n_substamps; j++) {
            PyObject *substamp_tuple = PyList_GetItem(substamps_list, j);
            stamps[i].xss[j] = (int)PyFloat_AsDouble(PyTuple_GetItem(substamp_tuple, 0));
            stamps[i].yss[j] = (int)PyFloat_AsDouble(PyTuple_GetItem(substamp_tuple, 1));
        }
        
        if (stamps[i].nss > 0) {
            fillStamp(state, &stamps[i], (float *)PyArray_DATA(conv_arr), (float *)PyArray_DATA(ref_arr));
        }
    }

    // 2. Iterative fitting loop (adapted from original fitKernel)
    double *kernel_sol = (double *)calloc(state->nCompTotal + 1, sizeof(double));
    if (!kernel_sol) {
        freeStampMem(state, stamps, n_stamps);
        free(stamps);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate kernel_sol");
        return NULL;
    }

    double meansig_substamps = 0.0, scatter_substamps = 0.0;
    int n_skipped_substamps = 0;
    
    // Ensure kernel vectors are initialized before the loop
    getKernelVec(state);

    fitKernel(state, stamps, (float *)PyArray_DATA(ref_arr), (float *)PyArray_DATA(conv_arr),
              (float *)PyArray_DATA(noise_arr), kernel_sol,
              &meansig_substamps, &scatter_substamps, &n_skipped_substamps);

    // 3. Prepare return values
    npy_intp dims[] = {state->nCompTotal + 1};
    double *kernel_sol_copy = (double *)malloc((state->nCompTotal + 1) * sizeof(double));
    memcpy(kernel_sol_copy, kernel_sol, (state->nCompTotal + 1) * sizeof(double));
    PyObject *global_coeffs_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, kernel_sol_copy);
    PyArray_ENABLEFLAGS((PyArrayObject *)global_coeffs_array, NPY_ARRAY_OWNDATA);
    
    PyObject *final_stamps_list = PyList_New(0);
    for (int i = 0; i < n_stamps; i++)
    {
        if (stamps[i].sscnt < stamps[i].nss)
        {
            PyObject *stamp_dict = PyDict_New();
            PyDict_SetItemString(stamp_dict, "x", PyFloat_FromDouble(stamps[i].xss[stamps[i].sscnt]));
            PyDict_SetItemString(stamp_dict, "y", PyFloat_FromDouble(stamps[i].yss[stamps[i].sscnt]));
            PyDict_SetItemString(stamp_dict, "figure_of_merit", PyFloat_FromDouble(stamps[i].chi2));
            PyList_Append(final_stamps_list, stamp_dict);
        }
    }

    PyObject *stats_dict = PyDict_New();
    PyDict_SetItemString(stats_dict, "meansig", PyFloat_FromDouble(meansig_substamps));
    PyDict_SetItemString(stats_dict, "scatter", PyFloat_FromDouble(scatter_substamps));
    PyDict_SetItemString(stats_dict, "nskipped", PyLong_FromLong(n_skipped_substamps));

    // // 4. Cleanup
    // freeStampMem(state, stamps, n_stamps);
    // free(stamps);
    // free(kernel_sol); 

    return Py_BuildValue("OOO", global_coeffs_array, final_stamps_list, stats_dict);
}

static PyObject *py_apply_kernel(PyObject *self, PyObject *args)
{
    PyArrayObject *image_arr, *kernel_solution_arr;
    PyArrayObject *noise_sq_arr;
    HotpantsStateObject *state_obj;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", &HotpantsState_Type, &state_obj, &PyArray_Type, &image_arr, &PyArray_Type, &kernel_solution_arr, &PyArray_Type, &noise_sq_arr))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    // Allocate temp arrays for this function
    int *mRData_out = (int *)calloc(state->nx * state->ny, sizeof(int));
    if (!mRData_out)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate mask");
        return NULL;
    }
    
    // Use the state's pre-allocated buffers, don't re-allocate them.
    // The main deallocator will handle freeing them.
    getKernelVec(state);

    double *kernel_sol = (double *)PyArray_DATA(kernel_solution_arr);
    float *conv_image = (float *)calloc(state->nx * state->ny, sizeof(float));
    if (!conv_image)
    {
        free(mRData_out);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate image data");
        return NULL;
    }

    // ========================================================================
    // START FIX: Correctly handle memory for the noise/variance image
    // ========================================================================
    long num_pixels = state->nx * state->ny;
    float *noise_sq_data_c_copy = (float *)malloc(num_pixels * sizeof(float));
    if (!noise_sq_data_c_copy) {
        free(conv_image);
        free(mRData_out);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate C-copy of noise data");
        return NULL;
    }
    memcpy(noise_sq_data_c_copy, (float *)PyArray_DATA(noise_sq_arr), num_pixels * sizeof(float));

    // Set the state's mRData to our output buffer for this operation
    state->mRData = mRData_out;

    spatial_convolve(state, (float *)PyArray_DATA(image_arr), &noise_sq_data_c_copy, state->nx, state->ny, kernel_sol, conv_image, state->mRData);
    
    // Unset the state's mRData pointer so it doesn't point to local memory anymore
    state->mRData = NULL;
    // ========================================================================
    // END FIX
    // ========================================================================


    npy_intp dims[] = {state->ny, state->nx};
    PyObject *conv_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, conv_image);
    if (!conv_arr)
    {
        free(conv_image);
        free(mRData_out);
        free(noise_sq_data_c_copy); // Clean up on error
        PyErr_SetString(PyExc_MemoryError, "Failed to create conv_arr");
        return NULL;
    }
    PyObject *mask_arr = PyArray_SimpleNewFromData(2, dims, NPY_INT32, mRData_out);
    if (!mask_arr)
    {
        free(conv_image);
        free(mRData_out);
        free(noise_sq_data_c_copy); // Clean up on error
        PyErr_SetString(PyExc_MemoryError, "Failed to create mask_arr");
        return NULL;
    }

    // Create an object for the convolved noise image from the C-managed buffer.
    PyObject *conv_noise_sq_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, noise_sq_data_c_copy);
    if (!conv_noise_sq_arr)
    {
        free(conv_image);
        free(mRData_out);
        free(noise_sq_data_c_copy); // Clean up on error
        PyErr_SetString(PyExc_MemoryError, "Failed to create conv_noise_sq_arr");
        return NULL;
    }

    Py_INCREF(conv_arr);
    Py_INCREF(mask_arr);
    Py_INCREF(conv_noise_sq_arr);

    // Transfer ownership of all C-allocated buffers to their respective NumPy arrays.
    // NumPy will now be responsible for calling `free()` when the arrays are garbage collected.
    PyArray_ENABLEFLAGS((PyArrayObject *)conv_arr, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject *)mask_arr, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject *)conv_noise_sq_arr, NPY_ARRAY_OWNDATA);

    return Py_BuildValue("OOO", conv_arr, mask_arr, conv_noise_sq_arr);
}

static PyObject *py_get_background_image(PyObject *self, PyObject *args)
{
    PyArrayObject *kernel_sol_arr;
    HotpantsStateObject *state_obj;
    if (!PyArg_ParseTuple(args, "O!O!", &HotpantsState_Type, &state_obj, &PyArray_Type, &kernel_sol_arr))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    double *kernel_sol = (double *)PyArray_DATA(kernel_sol_arr);
    float *bkg_image = (float *)calloc(state->nx * state->ny, sizeof(float));
    if (!bkg_image)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate bkg_image");
        return NULL;
    }

    for (int j = 0; j < state->ny; ++j)
    {
        for (int i = 0; i < state->nx; ++i)
        {
            bkg_image[i + state->nx * j] = get_background(state, i, j, kernel_sol);
        }
    }
    npy_intp dims[] = {state->ny, state->nx};
    PyObject *bkg_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, bkg_image);
    if (!bkg_arr)
    {
        free(bkg_image);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array from data");
        return NULL;
    }
    Py_INCREF(bkg_arr);

    PyArray_ENABLEFLAGS((PyArrayObject *)bkg_arr, NPY_ARRAY_OWNDATA);

    return bkg_arr;
}

static PyObject *py_rescale_noise_ok(PyObject *self, PyObject *args)
{
    PyArrayObject *diff_arr, *noise_arr, *mask_arr;
    HotpantsStateObject *state_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &HotpantsState_Type, &state_obj, &PyArray_Type, &diff_arr, &PyArray_Type, &noise_arr, &PyArray_Type, &mask_arr))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    float *diff_data = (float *)PyArray_DATA(diff_arr);
    float *noise_data = (float *)PyArray_DATA(noise_arr);
    int *mask_data = (int *)PyArray_DATA(mask_arr);
    double sdm, nmeanm;

    // Allocate a local state for the stats functions as they expect it
    hotpants_state_t local_state;
    memset(&local_state, 0, sizeof(hotpants_state_t));
    local_state.rPixX = state->nx;
    local_state.rPixY = state->ny;
    local_state.mRData = mask_data;
    local_state.statSig = state->statSig;
    local_state.fillVal = state->fillVal;

    getStampStats3(&local_state, diff_data, 0, 0, state->nx, state->ny, NULL, NULL, NULL, NULL, &sdm, NULL, NULL, 0x0, FLAG_OUTPUT_ISBAD, 5);
    getStampStats3(&local_state, noise_data, 0, 0, state->nx, state->ny, NULL, &nmeanm, NULL, NULL, NULL, NULL, NULL, 0x0, FLAG_OUTPUT_ISBAD, 5);
    double diff_rat = sdm / nmeanm;

    float *new_noise_data = (float *)calloc(state->nx * state->ny, sizeof(float));
    if (!new_noise_data)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate new_noise_data");
        return NULL;
    }

    for (int i = 0; i < state->nx * state->ny; i++)
    {
        new_noise_data[i] = noise_data[i];
        if (diff_rat > 1 && (mask_data[i] & 0xff) && (!(mask_data[i] & FLAG_OUTPUT_ISBAD)))
        {
            new_noise_data[i] *= diff_rat;
        }
    }
    npy_intp dims[] = {state->ny, state->nx};
    PyObject *new_noise_arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, new_noise_data);
    if (!new_noise_arr)
    {
        free(new_noise_data);
        PyErr_SetString(PyExc_MemoryError, "Failed to create NumPy array from data");
        return NULL;
    }
    Py_INCREF(new_noise_arr);
    PyArray_ENABLEFLAGS((PyArrayObject *)new_noise_arr, NPY_ARRAY_OWNDATA);

    return new_noise_arr;
}

static PyObject *py_calculate_final_stats(PyObject *self, PyObject *args)
{
    PyArrayObject *diff_arr, *noise_arr, *mask_arr;
    HotpantsStateObject *state_obj;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &HotpantsState_Type, &state_obj, &PyArray_Type, &diff_arr, &PyArray_Type, &noise_arr, &PyArray_Type, &mask_arr))
    {
        return NULL;
    }
    hotpants_state_t *state = state_obj->state;

    float *diff_data = (float *)PyArray_DATA(diff_arr);
    float *noise_data = (float *)PyArray_DATA(noise_arr);
    int *mask_data = (int *)PyArray_DATA(mask_arr);

    hotpants_state_t local_state;
    memset(&local_state, 0, sizeof(hotpants_state_t));
    local_state.rPixX = state->nx;
    local_state.rPixY = state->ny;
    local_state.mRData = mask_data;
    local_state.statSig = state->statSig;
    local_state.fillVal = state->fillVal;

    double sum, mean, median, mode, sd, fwhm, lfwhm;
    double nsum, nmean, nmedian, nmode, nsd, nfwhm, nlfwhm;
    double x2norm;
    int nx2norm;

    getStampStats3(&local_state, diff_data, 0, 0, state->nx, state->ny, &sum, &mean, &median, &mode, &sd, &fwhm, &lfwhm, 0x0, 0xffff, 5);
    getStampStats3(&local_state, noise_data, 0, 0, state->nx, state->ny, &nsum, &nmean, &nmedian, &nmode, &nsd, &nfwhm, &nlfwhm, 0x0, 0xffff, 5);
    getNoiseStats3(&local_state, diff_data, noise_data, &x2norm, &nx2norm, 0x0, 0xffff);

    PyObject *stats_dict = PyDict_New();
    PyDict_SetItemString(stats_dict, "diff_mean", PyFloat_FromDouble(mean));
    PyDict_SetItemString(stats_dict, "diff_std", PyFloat_FromDouble(sd));
    PyDict_SetItemString(stats_dict, "noise_mean", PyFloat_FromDouble(nmean));
    PyDict_SetItemString(stats_dict, "x2norm", PyFloat_FromDouble(x2norm));
    PyDict_SetItemString(stats_dict, "nx2norm", PyLong_FromLong(nx2norm));
    return stats_dict;
}

// Helper function definitions
static int parse_config_dict(PyObject *config_dict, hotpants_config_t *config)
{
    if (!PyDict_Check(config_dict))
    {
        PyErr_SetString(PyExc_TypeError, "Configuration must be a dictionary");
        return -1;
    }
    config->tuthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tuthresh"));
    PyObject *tuktresh_obj = PyDict_GetItemString(config_dict, "tuktresh");
    config->tuktresh = (tuktresh_obj == Py_None) ? config->tuthresh : PyFloat_AsDouble(tuktresh_obj);
    config->tlthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tlthresh"));
    config->tgain = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tgain"));
    config->trdnoise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "trdnoise"));
    config->tpedestal = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "tpedestal"));
    config->iuthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "iuthresh"));
    PyObject *iuktresh_obj = PyDict_GetItemString(config_dict, "iuktresh");
    config->iuktresh = (iuktresh_obj == Py_None) ? config->iuthresh : PyFloat_AsDouble(iuktresh_obj);
    config->ilthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ilthresh"));
    config->igain = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "igain"));
    config->irdnoise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "irdnoise"));
    config->ipedestal = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ipedestal"));
    config->rkernel = PyLong_AsLong(PyDict_GetItemString(config_dict, "rkernel"));
    config->ko = PyLong_AsLong(PyDict_GetItemString(config_dict, "ko"));
    config->bgo = PyLong_AsLong(PyDict_GetItemString(config_dict, "bgo"));
    config->fitthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fitthresh"));
    config->scale_fitthresh = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "scale_fitthresh"));
    config->min_frac_stamps = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "min_frac_stamps"));
    config->nss = PyLong_AsLong(PyDict_GetItemString(config_dict, "nss"));
    config->rss = PyLong_AsLong(PyDict_GetItemString(config_dict, "rss"));
    config->ks = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "ks"));
    config->kfm = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "kfm"));
    config->verbose = PyLong_AsLong(PyDict_GetItemString(config_dict, "verbose"));
    const char *force_conv_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "force_convolve"));
    strncpy(config->force_convolve, force_conv_str, 1);
    config->force_convolve[1] = '\0';
    const char *normalize_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "normalize"));
    strncpy(config->normalize, normalize_str, 1);
    config->normalize[1] = '\0';
    const char *fom_str = PyUnicode_AsUTF8(PyDict_GetItemString(config_dict, "fom"));
    strncpy(config->fom, fom_str, 1);
    config->fom[1] = '\0';
    config->fillval = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fillval"));
    config->fillval_noise = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "fillval_noise"));
    config->rescale_ok = PyLong_AsLong(PyDict_GetItemString(config_dict, "rescale_ok"));
    config->conv_var = PyLong_AsLong(PyDict_GetItemString(config_dict, "conv_var"));
    config->use_pca = PyLong_AsLong(PyDict_GetItemString(config_dict, "use_pca"));
    config->nregx = PyLong_AsLong(PyDict_GetItemString(config_dict, "nregx"));
    config->nregy = PyLong_AsLong(PyDict_GetItemString(config_dict, "nregy"));
    config->nstampx = PyLong_AsLong(PyDict_GetItemString(config_dict, "nstampx"));
    config->nstampy = PyLong_AsLong(PyDict_GetItemString(config_dict, "nstampy"));
    config->ngauss = PyLong_AsLong(PyDict_GetItemString(config_dict, "ngauss"));
    PyObject *deg_list = PyDict_GetItemString(config_dict, "deg_fixe");
    if (deg_list)
    {
        config->deg_fixe = (int *)malloc(config->ngauss * sizeof(int));
        for (int i = 0; i < config->ngauss; ++i)
            config->deg_fixe[i] = PyLong_AsLong(PyList_GetItem(deg_list, i));
    }
    PyObject *sig_list = PyDict_GetItemString(config_dict, "sigma_gauss");
    if (sig_list)
    {
        config->sigma_gauss = (float *)malloc(config->ngauss * sizeof(float));
        for (int i = 0; i < config->ngauss; ++i)
            config->sigma_gauss[i] = PyFloat_AsDouble(PyList_GetItem(sig_list, i));
    }
    config->fwkernel = PyLong_AsLong(PyDict_GetItemString(config_dict, "fwkernel"));
    config->fwksstamp = PyLong_AsLong(PyDict_GetItemString(config_dict, "fwksstamp"));
    config->ncomp_ker = PyLong_AsLong(PyDict_GetItemString(config_dict, "ncomp_ker"));
    config->ncomp = PyLong_AsLong(PyDict_GetItemString(config_dict, "ncomp"));
    config->n_bg_vectors = PyLong_AsLong(PyDict_GetItemString(config_dict, "n_bg_vectors"));
    config->n_comp_total = PyLong_AsLong(PyDict_GetItemString(config_dict, "n_comp_total"));
    config->stat_sig = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "stat_sig"));
    config->kf_spread_mask1 = PyFloat_AsDouble(PyDict_GetItemString(config_dict, "kf_spread_mask1"));
    return 0;
}

static void init_hotpants_state(hotpants_state_t *state, int nx, int ny, const hotpants_config_t *config)
{
    memset(state, 0, sizeof(hotpants_state_t));
    state->nx = nx;
    state->ny = ny;
    state->rPixX = nx;
    state->rPixY = ny;
    state->tUThresh = config->tuthresh;
    state->tLThresh = config->tlthresh;
    state->tUKThresh = config->tuktresh;
    state->tGain = config->tgain;
    state->tRdnoise = config->trdnoise;
    state->tPedestal = config->tpedestal;
    state->iUThresh = config->iuthresh;
    state->iLThresh = config->ilthresh;
    state->iUKThresh = config->iuktresh;
    state->iGain = config->igain;
    state->iRdnoise = config->irdnoise;
    state->iPedestal = config->ipedestal;
    state->hwKernel = config->rkernel;
    state->kerOrder = config->ko;
    state->bgOrder = config->bgo;
    state->kerFitThresh = config->fitthresh;
    state->kerSigReject = config->ks;
    state->kerFracMask = config->kfm;
    state->nKSStamps = config->nss;
    state->hwKSStamp = config->rss;
    state->nRegX = config->nregx;
    state->nRegY = config->nregy;
    state->nStampX = config->nstampx;
    state->nStampY = config->nstampy;
    strncpy(state->forceConvolve_str, config->force_convolve, 2);
    strncpy(state->photNormalize_str, config->normalize, 2);
    strncpy(state->figMerit_str, config->fom, 2);
    state->fillVal = config->fillval;
    state->fillValNoise = config->fillval_noise;
    state->verbose = config->verbose;
    state->rescaleOK = config->rescale_ok;
    state->convolveVariance = config->conv_var;
    state->usePCA = config->use_pca;
    state->statSig = config->stat_sig;
    state->kfSpreadMask1 = config->kf_spread_mask1;
    state->ngauss = config->ngauss;
    state->deg_fixe = (int *)malloc(state->ngauss * sizeof(int));
    memcpy(state->deg_fixe, config->deg_fixe, state->ngauss * sizeof(int));
    state->sigma_gauss = (float *)malloc(state->ngauss * sizeof(float));
    memcpy(state->sigma_gauss, config->sigma_gauss, state->ngauss * sizeof(float));
    state->fwKernel = config->fwkernel;
    state->fwStamp = 2 * state->hwKernel + 1;
    state->hwStamp = state->hwKernel;
    state->fwKSStamp = config->fwksstamp;
    state->nStamps = state->nStampX * state->nStampY;
    state->nCompKer = config->ncomp_ker;
    state->nComp = config->ncomp;
    state->nBGVectors = config->n_bg_vectors;
    state->nCompTotal = config->n_comp_total;
    state->nC = state->nCompKer + state->nBGVectors + 1;
    state->kcStep = state->fwKernel;

    // Allocate memory for filters, kernels, and temporary arrays as in main.c
    state->temp = (float *)calloc((state->fwKSStamp + state->fwKernel) * state->fwKSStamp, sizeof(float));
    state->indx = (int *)calloc(state->nCompTotal + 1 + 100, sizeof(int)); // Add 100 as in main.c for safety
    state->kernel = (double *)calloc(state->fwKernel * state->fwKernel, sizeof(double));
    state->kernel_coeffs = (double *)calloc(state->nCompKer, sizeof(double));
    state->kernel_vec = (double **)calloc(state->nCompKer, sizeof(double *));
    state->filter_x = (double *)calloc(state->nCompKer * state->fwKernel, sizeof(double));
    state->filter_y = (double *)calloc(state->nCompKer * state->fwKernel, sizeof(double));
    state->check_mat = (double **)calloc(state->nC, sizeof(double *));
    for (int i = 0; i < state->nC; i++)
    {
        state->check_mat[i] = (double *)calloc(state->nC, sizeof(double));
    }
    state->check_vec = (double *)calloc(state->nC, sizeof(double));
    state->check_stack = (double *)calloc(state->nStamps, sizeof(double));

    // Allocate memory for PCA vectors if use_pca is enabled
    if (state->usePCA)
    {
        state->PCA = (double **)calloc(state->nCompKer, sizeof(double *));
        for (int i = 0; i < state->nCompKer; ++i)
        {
            state->PCA[i] = (double *)calloc(state->fwKernel * state->fwKernel, sizeof(double));
        }
    }
}

static void free_hotpants_state(HotpantsStateObject *self)
{
    hotpants_state_t *state = self->state;
    if (state)
    {
        // Free all memory owned by the C state object
        if (state->mRData)
            free(state->mRData);
        if (state->temp)
            free(state->temp);
        if (state->temp2)
            free(state->temp2);
        if (state->indx)
            free(state->indx);
        if (state->kernel)
            free(state->kernel);
        if (state->kernel_coeffs)
            free(state->kernel_coeffs);
        if (state->kernel_vec)
        {
            for (int i = 0; i < state->nCompKer; ++i)
                if (state->kernel_vec[i])
                    free(state->kernel_vec[i]);
            free(state->kernel_vec);
        }
        if (state->filter_x)
            free(state->filter_x);
        if (state->filter_y)
            free(state->filter_y);
        if (state->check_mat)
        {
            for (int i = 0; i < state->nC; ++i)
                if (state->check_mat[i])
                    free(state->check_mat[i]);
            free(state->check_mat);
        }
        if (state->check_vec)
            free(state->check_vec);
        if (state->check_stack)
            free(state->check_stack);
        if (state->wxy)
        {
            for (int i = 0; i < state->nS; ++i)
                if (state->wxy[i])
                    free(state->wxy[i]);
            free(state->wxy);
        }
        if (state->PCA)
        {
            for (int i = 0; i < state->ngauss; ++i)
                if (state->PCA[i])
                    free(state->PCA[i]);
            free(state->PCA);
        }
        if (state->tStamps)
        {
            freeStampMem(state, state->tStamps, state->nStamps);
            free(state->tStamps);
        }
        if (state->iStamps)
        {
            freeStampMem(state, state->iStamps, state->nStamps);
            free(state->iStamps);
        }
        if (state->deg_fixe)
            free(state->deg_fixe);
        if (state->sigma_gauss)
            free(state->sigma_gauss);

        free(state);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyMethodDef hotpants_ext_methods[] = {
    {"make_input_mask", py_make_input_mask, METH_VARARGS, "Creates a pixel mask based on input data and thresholds."},
    {"calculate_noise_image", py_calculate_noise_image, METH_VARARGS, "Calculates a noise image based on gain and readnoise."},
    {"find_stamps", py_find_stamps, METH_VARARGS, "Finds potential stamps for kernel fitting."},
    {"fit_stamps_and_get_fom", py_fit_stamps_and_get_fom, METH_VARARGS, "Fits individual stamps and returns figures of merit."},
    {"fit_kernel", (PyCFunction)py_fit_kernel, METH_VARARGS, "Performs iterative kernel fitting and solves for coefficients."},
    {"apply_kernel", py_apply_kernel, METH_VARARGS, "Applies a global kernel solution to convolve an image."},
    {"get_background_image", py_get_background_image, METH_VARARGS, "Calculates the background image from the kernel solution."},
    {"rescale_noise_ok", py_rescale_noise_ok, METH_VARARGS, "Rescales noise for 'OK' pixels to match empirical noise ratio."},
    {"calculate_final_stats", py_calculate_final_stats, METH_VARARGS, "Calculates final statistics on the difference image."},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef hotpants_ext_module = {
    PyModuleDef_HEAD_INIT, "hotpants_ext", "HOTPanTS C extension for image differencing", -1, hotpants_ext_methods};

PyMODINIT_FUNC PyInit_hotpants_ext(void)
{
    PyObject *module;

    // Finalize the type object
    HotpantsState_Type.tp_new = hotpants_state_new;
    HotpantsState_Type.tp_dealloc = (destructor)hotpants_state_dealloc;
    HotpantsState_Type.tp_init = (initproc)hotpants_state_init_from_config;
    if (PyType_Ready(&HotpantsState_Type) < 0)
    {
        return NULL;
    }

    module = PyModule_Create(&hotpants_ext_module);
    if (module == NULL)
    {
        return NULL;
    }

    // Add the type object to the module
    Py_INCREF(&HotpantsState_Type);
    if (PyModule_AddObject(module, "HotpantsState", (PyObject *)&HotpantsState_Type) < 0)
    {
        Py_DECREF(&HotpantsState_Type);
        Py_DECREF(module);
        return NULL;
    }

    import_array();

    return module;
}
