/* NumPy boundary for the stateful HOTPANTS implementation.
 * Every public entry validates buffers before handing pointers to C.
 * Python owns result arrays; state and temporary C allocations have one owner.
 */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "functions.h"
#include "globals.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    float tuthresh, tlthresh, tuktresh, tgain, trdnoise, tpedestal;
    float iuthresh, ilthresh, iuktresh, igain, irdnoise, ipedestal;
    int rkernel, ko, bgo;
    float fitthresh, ks, kfm;
    int nss, rss, nregx, nregy, nstampx, nstampy;
    char force_convolve[2], normalize[2], fom[2];
    float fillval, fillval_noise;
    int verbose, rescale_ok, conv_var, use_pca, ngauss;
    int *deg_fixe;
    float *sigma_gauss;
    int fwkernel, fwksstamp, ncomp_ker, ncomp, n_bg_vectors, n_comp_total;
    float stat_sig, kf_spread_mask1;
} hotpants_config_t;
typedef struct
{
    PyObject_HEAD hotpants_state_t *state;
} HotpantsStateObject;
static PyTypeObject HotpantsState_Type;

static void release_state(hotpants_state_t *s)
{
    if (!s)
        return;
    free(s->mRData);
    free(s->temp);
    free(s->temp2);
    free(s->indx);
    free(s->kernel);
    free(s->kernel_coeffs);
    if (s->kernel_vec)
    {
        for (int i = 0; i < s->nCompKer; i++)
            free(s->kernel_vec[i]);
        free(s->kernel_vec);
    }
    free(s->filter_x);
    free(s->filter_y);
    if (s->check_mat)
    {
        for (int i = 0; i < s->nC; i++)
            free(s->check_mat[i]);
        free(s->check_mat);
    }
    free(s->check_vec);
    free(s->check_stack);
    if (s->wxy)
    {
        for (int i = 0; i < s->nS; i++)
            free(s->wxy[i]);
        free(s->wxy);
    }
    if (s->PCA)
    {
        for (int i = 0; i < s->nCompKer; i++)
            free(s->PCA[i]);
        free(s->PCA);
    }
    freeStampMem(s, s->tStamps, s->nStamps);
    free(s->tStamps);
    freeStampMem(s, s->iStamps, s->nStamps);
    free(s->iStamps);
    free(s->deg_fixe);
    free(s->sigma_gauss);
    free(s->xcmp);
    free(s->ycmp);
    free(s);
}
static void state_dealloc(HotpantsStateObject *self)
{
    release_state(self->state);
    Py_TYPE(self)->tp_free((PyObject *)self);
}
static PyObject *state_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    HotpantsStateObject *self = (HotpantsStateObject *)type->tp_alloc(type, 0);
    if (self)
        self->state = NULL;
    return (PyObject *)self;
}
static hotpants_state_t *begin(HotpantsStateObject *obj)
{
    if (!obj->state || !obj->state->mRData)
    {
        PyErr_SetString(PyExc_RuntimeError, "HotpantsState is not initialized");
        return NULL;
    }
    obj->state->error = 0;
    return obj->state;
}
static int native_ok(hotpants_state_t *s)
{
    if (!s->error)
        return 1;
    if (s->error == 1)
        PyErr_NoMemory();
    else
        PyErr_SetString(PyExc_ValueError, "HOTPANTS fit is singular or has invalid stamp support");
    return 0;
}
static int array_ok(PyArrayObject *a, int dtype, int ndim, const npy_intp *shape, const char *name)
{
    if (PyArray_TYPE(a) != dtype || PyArray_NDIM(a) != ndim || !PyArray_ISCARRAY_RO(a) ||
        !PyArray_ISNOTSWAPPED(a))
    {
        PyErr_Format(PyExc_ValueError,
                     "%s must have the required dtype, dimensions, native byte order and aligned "
                     "C-contiguous layout",
                     name);
        return 0;
    }
    for (int i = 0; i < ndim; i++)
        if (shape && shape[i] >= 0 && PyArray_DIM(a, i) != shape[i])
        {
            PyErr_Format(PyExc_ValueError, "%s has an incompatible shape", name);
            return 0;
        }
    return 1;
}
static int image_ok(hotpants_state_t *s, PyArrayObject *a, int dtype, const char *name)
{
    npy_intp dims[] = {s->ny, s->nx};
    return array_ok(a, dtype, 2, dims, name);
}
static int kernel_ok(hotpants_state_t *s, PyArrayObject *a)
{
    npy_intp dims[] = {s->nCompTotal + 1};
    if (!array_ok(a, NPY_DOUBLE, 1, dims, "kernel_solution"))
        return 0;
    double *v = PyArray_DATA(a);
    for (npy_intp i = 0; i < dims[0]; i++)
        if (!isfinite(v[i]))
        {
            PyErr_SetString(PyExc_ValueError, "kernel_solution must be finite");
            return 0;
        }
    return 1;
}
static int owned_item(PyObject *dict, const char *key, PyObject *value)
{
    if (!value)
        return 0;
    int ok = PyDict_SetItemString(dict, key, value) == 0;
    Py_DECREF(value);
    return ok;
}
static void buffer_free(PyObject *capsule)
{
    free(PyCapsule_GetPointer(capsule, "hotpants.buffer"));
}
static PyObject *wrap_buffer(void *data, int ndim, npy_intp *dims, int dtype)
{
    if (!data)
        return PyErr_NoMemory();
    PyObject *owner = PyCapsule_New(data, "hotpants.buffer", buffer_free);
    if (!owner)
    {
        free(data);
        return NULL;
    }
    PyObject *a = PyArray_SimpleNewFromData(ndim, dims, dtype, data);
    if (!a)
    {
        Py_DECREF(owner);
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)a, owner) < 0)
    {
        Py_DECREF(a);
        return NULL;
    }
    return a;
}
static double **matrix_new(int n)
{
    double **a = calloc(n, sizeof(*a));
    if (!a)
        return NULL;
    for (int i = 0; i < n; i++)
        if (!(a[i] = calloc(n, sizeof(double))))
        {
            for (int j = 0; j < i; j++)
                free(a[j]);
            free(a);
            return NULL;
        }
    return a;
}
static void matrix_free(double **a, int n)
{
    if (a)
    {
        for (int i = 0; i < n; i++)
            free(a[i]);
        free(a);
    }
}
/* Do not execute user conversion/equality callbacks while holding borrowed
   pointers into mutable stamp/config containers. */
static int plain_keys(PyObject *dict)
{
    if (!PyDict_Check(dict))
    {
        PyErr_SetString(PyExc_TypeError, "expected a dictionary");
        return 0;
    }
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(dict, &pos, &key, &value))
        if (!PyUnicode_CheckExact(key))
        {
            PyErr_SetString(PyExc_TypeError, "dictionary keys must be plain strings");
            return 0;
        }
    return 1;
}
static int plain_number(PyObject *obj)
{
    if (obj && (PyLong_Check(obj) || PyFloat_Check(obj)))
        return 1;
    PyErr_SetString(PyExc_TypeError, "native configuration requires Python int/float values");
    return 0;
}
static int coordinate_tuple(PyObject *tuple, int *x, int *y)
{
    if (!PyTuple_Check(tuple) || PyTuple_GET_SIZE(tuple) != 2 ||
        !PyLong_Check(PyTuple_GET_ITEM(tuple, 0)) || !PyLong_Check(PyTuple_GET_ITEM(tuple, 1)))
    {
        PyErr_SetString(PyExc_TypeError, "coordinates must be a pair of Python integers");
        return 0;
    }
    return PyArg_ParseTuple(tuple, "ii", x, y);
}
static int dict_int(PyObject *d, const char *key, int *out)
{
    if (!PyDict_Check(d))
    {
        PyErr_SetString(PyExc_TypeError, "stamp entries must be dictionaries");
        return 0;
    }
    if (!plain_keys(d))
        return 0;
    PyObject *o = PyDict_GetItemString(d, key);
    if (!o)
    {
        PyErr_Format(PyExc_ValueError, "missing stamp key: %s", key);
        return 0;
    }
    if (!PyLong_Check(o))
    {
        PyErr_SetString(PyExc_TypeError, "stamp coordinates must be Python integers");
        return 0;
    }
    long v = PyLong_AsLong(o);
    if (PyErr_Occurred())
        return 0;
    if (v < INT_MIN || v > INT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "stamp integer is out of range");
        return 0;
    }
    *out = (int)v;
    return 1;
}
static int coords_ok(hotpants_state_t *s, int x, int y)
{
    int pad = s->hwKernel + s->hwKSStamp;
    if (x < pad || y < pad || x >= s->nx - pad || y >= s->ny - pad)
    {
        PyErr_SetString(PyExc_ValueError, "stamp lies outside the image's convolution support");
        return 0;
    }
    return 1;
}
static int config_get_required_float(PyObject *config_dict, const char *key, float *out)
{
    PyObject *obj = PyDict_GetItemString(config_dict, key);
    if (obj == NULL)
    {
        PyErr_Format(PyExc_KeyError, "Missing config key: %s", key);
        return -1;
    }
    if (obj == Py_None)
    {
        PyErr_Format(PyExc_ValueError, "Config key %s cannot be None", key);
        return -1;
    }
    if (!plain_number(obj))
        return -1;
    *out = (float)PyFloat_AsDouble(obj);
    if (PyErr_Occurred())
        return -1;
    return 0;
}

static int config_get_required_long(PyObject *config_dict, const char *key, int *out)
{
    PyObject *obj = PyDict_GetItemString(config_dict, key);
    if (obj == NULL)
    {
        PyErr_Format(PyExc_KeyError, "Missing config key: %s", key);
        return -1;
    }
    if (obj == Py_None)
    {
        PyErr_Format(PyExc_ValueError, "Config key %s cannot be None", key);
        return -1;
    }
    if (!PyLong_Check(obj))
    {
        PyErr_SetString(PyExc_TypeError, "configuration count must be a Python integer");
        return -1;
    }
    long value = PyLong_AsLong(obj);
    if (value < INT_MIN || value > INT_MAX)
    {
        PyErr_Format(PyExc_ValueError, "%s is out of range", key);
        return -1;
    }
    *out = (int)value;
    if (PyErr_Occurred())
        return -1;
    return 0;
}

static int config_get_required_char(PyObject *config_dict, const char *key, char *out)
{
    PyObject *obj = PyDict_GetItemString(config_dict, key);
    if (obj == NULL)
    {
        PyErr_Format(PyExc_KeyError, "Missing config key: %s", key);
        return -1;
    }
    if (!PyUnicode_Check(obj))
    {
        PyErr_Format(PyExc_TypeError, "Config key %s must be a string", key);
        return -1;
    }
    const char *str = PyUnicode_AsUTF8(obj);
    if (str == NULL)
        return -1;
    if (strlen(str) != 1)
    {
        PyErr_Format(PyExc_ValueError, "%s must be one character", key);
        return -1;
    }
    out[0] = str[0];
    out[1] = '\0';
    return 0;
}

static int parse_config_dict(PyObject *config_dict, hotpants_config_t *config)
{
    if (!PyDict_Check(config_dict))
    {
        PyErr_SetString(PyExc_TypeError, "Configuration must be a dictionary");
        return -1;
    }
    if (!plain_keys(config_dict))
        return -1;
    memset(config, 0, sizeof(*config));
    config->deg_fixe = NULL;
    config->sigma_gauss = NULL;

    if (config_get_required_float(config_dict, "tuthresh", &config->tuthresh) < 0)
        return -1;
    PyObject *tuktresh_obj = PyDict_GetItemString(config_dict, "tuktresh");
    if (tuktresh_obj == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing config key: tuktresh");
        return -1;
    }
    if (tuktresh_obj != Py_None && !plain_number(tuktresh_obj))
        return -1;
    config->tuktresh =
        (tuktresh_obj == Py_None) ? config->tuthresh : (float)PyFloat_AsDouble(tuktresh_obj);
    if (PyErr_Occurred())
        return -1;
    if (config_get_required_float(config_dict, "tlthresh", &config->tlthresh) < 0)
        return -1;
    if (config_get_required_float(config_dict, "tgain", &config->tgain) < 0)
        return -1;
    if (config_get_required_float(config_dict, "trdnoise", &config->trdnoise) < 0)
        return -1;
    if (config_get_required_float(config_dict, "tpedestal", &config->tpedestal) < 0)
        return -1;
    if (config_get_required_float(config_dict, "iuthresh", &config->iuthresh) < 0)
        return -1;
    PyObject *iuktresh_obj = PyDict_GetItemString(config_dict, "iuktresh");
    if (iuktresh_obj == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "Missing config key: iuktresh");
        return -1;
    }
    if (iuktresh_obj != Py_None && !plain_number(iuktresh_obj))
        return -1;
    config->iuktresh =
        (iuktresh_obj == Py_None) ? config->iuthresh : (float)PyFloat_AsDouble(iuktresh_obj);
    if (PyErr_Occurred())
        return -1;
    if (config_get_required_float(config_dict, "ilthresh", &config->ilthresh) < 0)
        return -1;
    if (config_get_required_float(config_dict, "igain", &config->igain) < 0)
        return -1;
    if (config_get_required_float(config_dict, "irdnoise", &config->irdnoise) < 0)
        return -1;
    if (config_get_required_float(config_dict, "ipedestal", &config->ipedestal) < 0)
        return -1;
    if (config_get_required_long(config_dict, "rkernel", &config->rkernel) < 0)
        return -1;
    if (config_get_required_long(config_dict, "ko", &config->ko) < 0)
        return -1;
    if (config_get_required_long(config_dict, "bgo", &config->bgo) < 0)
        return -1;
    if (config_get_required_float(config_dict, "fitthresh", &config->fitthresh) < 0)
        return -1;
    if (config_get_required_long(config_dict, "nss", &config->nss) < 0)
        return -1;
    if (config_get_required_long(config_dict, "rss", &config->rss) < 0)
        return -1;
    if (config_get_required_float(config_dict, "ks", &config->ks) < 0)
        return -1;
    if (config_get_required_float(config_dict, "kfm", &config->kfm) < 0)
        return -1;
    if (config_get_required_long(config_dict, "verbose", &config->verbose) < 0)
        return -1;
    if (config_get_required_char(config_dict, "force_convolve", config->force_convolve) < 0)
        return -1;
    if (config_get_required_char(config_dict, "normalize", config->normalize) < 0)
        return -1;
    if (config_get_required_char(config_dict, "fom", config->fom) < 0)
        return -1;
    if (config_get_required_float(config_dict, "fillval", &config->fillval) < 0)
        return -1;
    if (config_get_required_float(config_dict, "fillval_noise", &config->fillval_noise) < 0)
        return -1;
    if (config_get_required_long(config_dict, "rescale_ok", &config->rescale_ok) < 0)
        return -1;
    if (config_get_required_long(config_dict, "conv_var", &config->conv_var) < 0)
        return -1;
    if (config_get_required_long(config_dict, "use_pca", &config->use_pca) < 0)
        return -1;
    if (config_get_required_long(config_dict, "nregx", &config->nregx) < 0)
        return -1;
    if (config_get_required_long(config_dict, "nregy", &config->nregy) < 0)
        return -1;
    if (config_get_required_long(config_dict, "nstampx", &config->nstampx) < 0)
        return -1;
    if (config_get_required_long(config_dict, "nstampy", &config->nstampy) < 0)
        return -1;
    if (config_get_required_long(config_dict, "ngauss", &config->ngauss) < 0)
        return -1;
    PyObject *deg_list = PyDict_GetItemString(config_dict, "deg_fixe");
    PyObject *sig_list = PyDict_GetItemString(config_dict, "sigma_gauss");
    if (config->ngauss > 0)
    {
        if (!deg_list || !PyList_Check(deg_list))
        {
            PyErr_SetString(PyExc_ValueError, "deg_fixe must be a list when ngauss > 0");
            return -1;
        }
        if (!sig_list || !PyList_Check(sig_list))
        {
            PyErr_SetString(PyExc_ValueError, "sigma_gauss must be a list when ngauss > 0");
            return -1;
        }
        if (PyList_Size(deg_list) != config->ngauss || PyList_Size(sig_list) != config->ngauss)
        {
            PyErr_SetString(PyExc_ValueError,
                            "deg_fixe and sigma_gauss must have length == ngauss");
            return -1;
        }
        config->deg_fixe = (int *)malloc(config->ngauss * sizeof(int));
        if (config->deg_fixe == NULL)
        {
            PyErr_NoMemory();
            return -1;
        }
        config->sigma_gauss = (float *)malloc(config->ngauss * sizeof(float));
        if (config->sigma_gauss == NULL)
        {
            free(config->deg_fixe);
            config->deg_fixe = NULL;
            PyErr_NoMemory();
            return -1;
        }
        for (int i = 0; i < config->ngauss; ++i)
        {
            PyObject *degree_obj = PyList_GetItem(deg_list, i);
            PyObject *sigma_obj = PyList_GetItem(sig_list, i);
            if (!degree_obj || !PyLong_Check(degree_obj) || !plain_number(sigma_obj))
            {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_TypeError, "Gaussian degree must be an integer");
                return -1;
            }
            long degree = PyLong_AsLong(degree_obj);
            if (!PyErr_Occurred() && (degree < 0 || degree > 64))
                PyErr_SetString(PyExc_ValueError, "Gaussian degree must be 0..64");
            config->deg_fixe[i] = (int)degree;
            if (PyErr_Occurred())
            {
                free(config->deg_fixe);
                free(config->sigma_gauss);
                config->deg_fixe = NULL;
                config->sigma_gauss = NULL;
                return -1;
            }
            config->sigma_gauss[i] = (float)PyFloat_AsDouble(PyList_GetItem(sig_list, i));
            if (PyErr_Occurred())
            {
                free(config->deg_fixe);
                free(config->sigma_gauss);
                config->deg_fixe = NULL;
                config->sigma_gauss = NULL;
                return -1;
            }
        }
    }
    if (config_get_required_long(config_dict, "fwkernel", &config->fwkernel) < 0)
        return -1;
    if (config_get_required_long(config_dict, "fwksstamp", &config->fwksstamp) < 0)
        return -1;
    if (config_get_required_long(config_dict, "ncomp_ker", &config->ncomp_ker) < 0)
        return -1;
    if (config_get_required_long(config_dict, "ncomp", &config->ncomp) < 0)
        return -1;
    if (config_get_required_long(config_dict, "n_bg_vectors", &config->n_bg_vectors) < 0)
        return -1;
    if (config_get_required_long(config_dict, "n_comp_total", &config->n_comp_total) < 0)
        return -1;
    if (config_get_required_float(config_dict, "stat_sig", &config->stat_sig) < 0)
        return -1;
    if (config_get_required_float(config_dict, "kf_spread_mask1", &config->kf_spread_mask1) < 0)
        return -1;
    return 0;
}

static int validate_config(hotpants_config_t *c, int nx, int ny)
{
    if (nx <= 0 || ny <= 0 || (int64_t)nx * ny > INT_MAX || c->rkernel < 0 || c->rkernel > 16383 ||
        c->rss < 1 || c->rss > 16383 || c->ko < 0 || c->ko > 64 || c->bgo < -1 || c->bgo > 64 ||
        c->nss < 1 || c->nss > INT_MAX / 2 || c->nregx != 1 || c->nregy != 1 || c->nstampx < 1 ||
        c->nstampy < 1 || (int64_t)c->nstampx * c->nstampy > INT_MAX || c->ngauss < 1 || c->use_pca)
    {
        PyErr_SetString(PyExc_ValueError,
                        "invalid dimensions/configuration; PCA bases are not supported");
        return 0;
    }
    int64_t nk = 0, ns = ((int64_t)c->ko + 1) * (c->ko + 2) / 2,
            nb = ((int64_t)c->bgo + 1) * (c->bgo + 2) / 2;
    for (int j = 0; j < c->ngauss; j++)
    {
        if (c->deg_fixe[j] < 0 || c->deg_fixe[j] > 64 || !isfinite(c->sigma_gauss[j]) ||
            c->sigma_gauss[j] <= 0)
        {
            PyErr_SetString(PyExc_ValueError,
                            "Gaussian degrees must be 0..64 and coefficients finite and positive");
            return 0;
        }
        nk += ((int64_t)c->deg_fixe[j] + 1) * (c->deg_fixe[j] + 2) / 2;
    }
    if (nk > INT_MAX || nk * ns + nb > INT_MAX - 101 || (nk + nb + 1) * (nk + nb + 1) > INT_MAX ||
        (int64_t)(2 * c->rss + 1) * (2 * c->rss + 2 * c->rkernel + 2) > INT_MAX ||
        nk * (2 * c->rkernel + 1) > INT_MAX || c->fwkernel != 2 * c->rkernel + 1 ||
        c->fwksstamp != 2 * c->rss + 1 || c->ncomp_ker != nk || c->ncomp != ns ||
        c->n_bg_vectors != nb || c->n_comp_total != nk * ns + nb)
    {
        PyErr_SetString(PyExc_ValueError, "inconsistent or oversized derived kernel dimensions");
        return 0;
    }
    float values[] = {
        c->tuthresh,       c->tlthresh, c->tuktresh, c->tgain,   c->trdnoise,      c->tpedestal,
        c->iuthresh,       c->ilthresh, c->iuktresh, c->igain,   c->irdnoise,      c->ipedestal,
        c->fitthresh,      c->ks,       c->kfm,      c->fillval, c->fillval_noise, c->stat_sig,
        c->kf_spread_mask1};
    for (size_t j = 0; j < sizeof(values) / sizeof(values[0]); j++)
        if (!isfinite(values[j]))
        {
            PyErr_SetString(PyExc_ValueError, "configuration numbers must be finite");
            return 0;
        }
    if (c->tgain <= 0 || c->igain <= 0 || c->trdnoise < 0 || c->irdnoise < 0 || c->fitthresh < 0 ||
        c->ks <= 0 || c->stat_sig <= 0 || c->kfm < 0 || c->kfm > 1 || c->kf_spread_mask1 < 0 ||
        (double)c->kf_spread_mask1 * c->rkernel > INT_MAX || !strchr("tib", c->force_convolve[0]) ||
        !c->force_convolve[0] || !strchr("tiu", c->normalize[0]) || !c->normalize[0] ||
        !strchr("vsh", c->fom[0]) || !c->fom[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "invalid gain, noise, clipping, mask or direction parameter");
        return 0;
    }
    return 1;
}
static int state_init(HotpantsStateObject *self, PyObject *args, PyObject *kwds)
{
    int nx, ny;
    PyObject *dict;
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "HotpantsState takes positional arguments");
        return -1;
    }
    if (self->state)
    {
        PyErr_SetString(PyExc_RuntimeError, "HotpantsState cannot be reinitialized");
        return -1;
    }
    if (!PyArg_ParseTuple(args, "iiO!", &nx, &ny, &PyDict_Type, &dict))
        return -1;
    hotpants_config_t config = {0};
    if (parse_config_dict(dict, &config) < 0 || !validate_config(&config, nx, ny))
    {
        free(config.deg_fixe);
        free(config.sigma_gauss);
        return -1;
    }
    hotpants_state_t *state = calloc(1, sizeof(*state));
    if (!state)
    {
        free(config.deg_fixe);
        free(config.sigma_gauss);
        PyErr_NoMemory();
        return -1;
    }
    state->nx = nx;
    state->ny = ny;
    state->rPixX = nx;
    state->rPixY = ny;
    state->tUThresh = config.tuthresh;
    state->tLThresh = config.tlthresh;
    state->tUKThresh = config.tuktresh;
    state->tGain = config.tgain;
    state->tRdnoise = config.trdnoise;
    state->tPedestal = config.tpedestal;
    state->iUThresh = config.iuthresh;
    state->iLThresh = config.ilthresh;
    state->iUKThresh = config.iuktresh;
    state->iGain = config.igain;
    state->iRdnoise = config.irdnoise;
    state->iPedestal = config.ipedestal;
    state->hwKernel = config.rkernel;
    state->kerOrder = config.ko;
    state->bgOrder = config.bgo;
    state->kerFitThresh = config.fitthresh;
    state->kerSigReject = config.ks;
    state->kerFracMask = config.kfm;
    state->nKSStamps = config.nss;
    state->hwKSStamp = config.rss;
    state->nRegX = config.nregx;
    state->nRegY = config.nregy;
    state->nStampX = config.nstampx;
    state->nStampY = config.nstampy;
    strncpy(state->forceConvolve_str, config.force_convolve, 2);
    strncpy(state->photNormalize_str, config.normalize, 2);
    strncpy(state->figMerit_str, config.fom, 2);
    state->fillVal = config.fillval;
    state->fillValNoise = config.fillval_noise;
    state->verbose = config.verbose;
    state->rescaleOK = config.rescale_ok;
    state->convolveVariance = config.conv_var;
    state->usePCA = config.use_pca;
    state->statSig = config.stat_sig;
    state->kfSpreadMask1 = config.kf_spread_mask1;
    state->ngauss = config.ngauss;
    state->deg_fixe = config.deg_fixe;
    state->sigma_gauss = config.sigma_gauss;
    state->fwKernel = config.fwkernel;
    state->fwKSStamp = config.fwksstamp;
    state->fwStamp = imin(nx / config.nstampx, ny / config.nstampy) - state->fwKernel;
    if (state->fwStamp % 2 == 0)
        state->fwStamp--;
    state->fwStamp = imax(state->fwStamp, state->fwKSStamp + state->fwKernel);
    state->hwStamp = state->fwStamp / 2;
    state->nStamps = config.nstampx * config.nstampy;
    state->nCompKer = config.ncomp_ker;
    state->nComp = config.ncomp;
    state->nBGVectors = config.n_bg_vectors;
    state->nCompTotal = config.n_comp_total;
    state->nC = state->nCompKer + state->nBGVectors + 1;
    state->kcStep = state->fwKernel;
#define ALLOC(field, count, type)                                                                  \
    do                                                                                             \
    {                                                                                              \
        state->field = calloc((size_t)(count), sizeof(type));                                      \
        if (!state->field)                                                                         \
            goto fail;                                                                             \
    } while (0)
    ALLOC(mRData, (size_t)nx * ny, int);
    ALLOC(temp, (size_t)(state->fwKSStamp + state->fwKernel) * state->fwKSStamp, float);
    ALLOC(indx, state->nCompTotal + 101, int);
    ALLOC(kernel, (size_t)state->fwKernel * state->fwKernel, double);
    ALLOC(kernel_coeffs, state->nCompKer, double);
    ALLOC(kernel_vec, state->nCompKer, double *);
    ALLOC(filter_x, (size_t)state->nCompKer * state->fwKernel, double);
    ALLOC(filter_y, (size_t)state->nCompKer * state->fwKernel, double);
    state->check_mat = matrix_new(state->nC);
    if (!state->check_mat)
        goto fail;
    ALLOC(check_vec, state->nC, double);
    ALLOC(check_stack, state->nStamps, double);
#undef ALLOC
    self->state = state;
    return 0;
fail:
    release_state(state);
    PyErr_NoMemory();
    return -1;
}
static PyObject *py_make_input_mask(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *t, *i;
    PyObject *tm, *im;
    if (!PyArg_ParseTuple(args, "O!O!O!OO", &HotpantsState_Type, &obj, &PyArray_Type, &t,
                          &PyArray_Type, &i, &tm, &im))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s)
        return NULL;
    if (!image_ok(s, t, NPY_FLOAT32, "template") || !image_ok(s, i, NPY_FLOAT32, "image"))
        return NULL;
    PyObject *masks[] = {tm, im};
    for (int j = 0; j < 2; j++)
        if (masks[j] != Py_None && (!PyArray_Check(masks[j]) ||
                                    !image_ok(s, (PyArrayObject *)masks[j], NPY_INT32, "mask")))
        {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_TypeError, "mask must be an int32 NumPy array or None");
            return NULL;
        }
    npy_intp dims[] = {s->ny, s->nx};
    PyObject *out = PyArray_ZEROS(2, dims, NPY_INT32, 0);
    if (!out)
        return NULL;
    int *m = PyArray_DATA((PyArrayObject *)out);
    size_t n = (size_t)s->nx * s->ny;
    for (int j = 0; j < 2; j++)
        if (masks[j] != Py_None)
        {
            int *v = PyArray_DATA((PyArrayObject *)masks[j]);
            for (size_t k = 0; k < n; k++)
                if (v[k] > 0)
                    m[k] |= FLAG_INPUT_MASK;
        }
    makeInputMask(s, PyArray_DATA(t), PyArray_DATA(i), m);
    int pad = s->hwKSStamp + s->hwKernel;
    for (int y = 0; y < s->ny; y++)
        for (int x = 0; x < s->nx; x++)
            if (x < pad || y < pad || x >= s->nx - pad || y >= s->ny - pad)
                m[x + s->nx * y] |= FLAG_T_BAD | FLAG_I_BAD;
    memcpy(s->mRData, m, n * sizeof(int));
    return out;
}
static PyObject *py_calculate_noise_image(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *a;
    int templ;
    if (!PyArg_ParseTuple(args, "O!O!p", &HotpantsState_Type, &obj, &PyArray_Type, &a, &templ))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s || !image_ok(s, a, NPY_FLOAT32, "image"))
        return NULL;
    float gain = templ ? s->tGain : s->iGain, rn = templ ? s->tRdnoise : s->iRdnoise;
    float *v = makeNoiseImage4(s, PyArray_DATA(a), 1 / gain, rn / gain);
    npy_intp dims[] = {s->ny, s->nx};
    return wrap_buffer(v, 2, dims, NPY_FLOAT32);
}
static PyObject *stamp_coordinates(stamp_struct *stamps, int n)
{
    PyObject *list = PyList_New(0);
    if (!list)
        return NULL;
    int id = 0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < stamps[j].nss; k++)
        {
            PyObject *d = Py_BuildValue("{s:i,s:i,s:i,s:i}", "substamp_id", id++, "stamp_group_id",
                                        j, "x", stamps[j].xss[k], "y", stamps[j].yss[k]);
            if (!d)
            {
                Py_DECREF(list);
                return NULL;
            }
            int ok = PyList_Append(list, d);
            Py_DECREF(d);
            if (ok < 0)
            {
                Py_DECREF(list);
                return NULL;
            }
        }
    return list;
}
static PyObject *py_find_stamps(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *t, *i;
    PyObject *cat;
    double threshold;
    if (!PyArg_ParseTuple(args, "O!O!O!dO", &HotpantsState_Type, &obj, &PyArray_Type, &t,
                          &PyArray_Type, &i, &threshold, &cat))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s)
        return NULL;
    if (!image_ok(s, t, NPY_FLOAT32, "template") || !image_ok(s, i, NPY_FLOAT32, "image"))
        return NULL;
    if (!isfinite(threshold) || threshold < 0 || threshold > FLT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "invalid fit threshold");
        return NULL;
    }
    float *xy = NULL;
    npy_intp ncat = 0, shape[] = {-1, 2};
    if (cat != Py_None)
    {
        if (!PyArray_Check(cat))
        {
            PyErr_SetString(PyExc_TypeError, "catalog must be a NumPy array or None");
            return NULL;
        }
        if (!array_ok((PyArrayObject *)cat, NPY_FLOAT32, 2, shape, "catalog"))
            return NULL;
        xy = PyArray_DATA((PyArrayObject *)cat);
        ncat = PyArray_DIM((PyArrayObject *)cat, 0);
        for (npy_intp j = 0; j < 2 * ncat; j++)
            if (!isfinite(xy[j]) || xy[j] < INT_MIN / 2 || xy[j] > INT_MAX / 2)
            {
                PyErr_SetString(PyExc_ValueError,
                                "catalog coordinates must be finite and representable");
                return NULL;
            }
    }
    s->kerFitThresh = (float)threshold;
    freeStampMem(s, s->tStamps, s->nStamps);
    free(s->tStamps);
    s->tStamps = NULL;
    freeStampMem(s, s->iStamps, s->nStamps);
    free(s->iStamps);
    s->iStamps = NULL;
    for (int y = 0; y < s->ny; y++)
        for (int x = 0; x < s->nx; x++)
        {
            s->mRData[x + s->nx * y] &= ~(FLAG_T_SKIP | FLAG_I_SKIP);
            if (x < s->hwKernel + s->hwKSStamp || y < s->hwKernel + s->hwKSStamp ||
                x >= s->nx - s->hwKernel - s->hwKSStamp || y >= s->ny - s->hwKernel - s->hwKSStamp)
                s->mRData[x + s->nx * y] |= FLAG_T_BAD | FLAG_I_BAD;
        }
    s->tStamps = calloc(s->nStamps, sizeof(stamp_struct));
    s->iStamps = calloc(s->nStamps, sizeof(stamp_struct));
    if (!s->tStamps || !s->iStamps || allocateStamps(s, s->tStamps, s->nStamps) ||
        allocateStamps(s, s->iStamps, s->nStamps))
        return PyErr_NoMemory();
    int nt = 0, ni = 0, pad = s->hwKernel + s->hwKSStamp;
    for (int gy = 0; gy < s->nStampY; gy++)
        for (int gx = 0; gx < s->nStampX; gx++)
        {
            int x0 = (int)((int64_t)gx * s->nx / s->nStampX),
                y0 = (int)((int64_t)gy * s->ny / s->nStampY);
            int x1 = imin(x0 + s->fwStamp - 1, s->nx - 1),
                y1 = imin(y0 + s->fwStamp - 1, s->ny - 1);
            if (x1 < x0 || y1 < y0)
                continue;
            if (cat != Py_None)
            {
                for (npy_intp j = 0; j < ncat; j++)
                {
                    int x = (int)lroundf(xy[2 * j]), y = (int)lroundf(xy[2 * j + 1]);
                    if (x < pad || y < pad || x >= s->nx - pad || y >= s->ny - pad)
                        continue;
                    if (x > x0 + s->hwKernel + 1 && x < x1 - s->hwKernel - 1 &&
                        y > y0 + s->hwKernel + 1 && y < y1 - s->hwKernel - 1)
                        buildStamps(s, x0, x1, y0, y1, &ni, &nt, 0, 0, 0, s->iStamps, s->tStamps,
                                    PyArray_DATA(i), PyArray_DATA(t), x, y);
                    if (!native_ok(s))
                        return NULL;
                }
            }
            else
            {
                buildStamps(s, x0, x1, y0, y1, &ni, &nt, 1, 0, 0, s->iStamps, s->tStamps,
                            PyArray_DATA(i), PyArray_DATA(t), 0, 0);
                if (!native_ok(s))
                    return NULL;
            }
            if (s->tStamps[nt].nss > 0)
                nt++;
            if (s->iStamps[ni].nss > 0)
                ni++;
            if (nt >= s->nStamps || ni >= s->nStamps)
                goto finished;
        }
finished:;
    PyObject *tl = stamp_coordinates(s->tStamps, nt), *il = stamp_coordinates(s->iStamps, ni);
    if (!tl || !il)
    {
        Py_XDECREF(tl);
        Py_XDECREF(il);
        return NULL;
    }
    return Py_BuildValue("NN", tl, il);
}
/* Create one self-contained result: no NumPy view points into a stamp/state. */
static PyObject *local_result(hotpants_state_t *s, stamp_struct *stamp, float *noise,
                              const char *dir, int id)
{
    int w = s->fwKSStamp, npix = w * w, nv = s->nCompKer + s->nBGVectors;
    int nc = s->nCompKer + (s->nBGVectors > 0);
    npy_intp dims[] = {w, w}, bdims[] = {nv, w, w}, ldims[] = {nc};
    PyObject *d = PyDict_New();
    double **mat = NULL, *sol = NULL;
    PyArrayObject *conv = NULL, *ref = NULL, *no = NULL, *basis = NULL, *solution = NULL,
                  *model = NULL;
    if (!d)
        goto fail;
    conv = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    ref = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    no = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    basis = (PyArrayObject *)PyArray_SimpleNew(3, bdims, NPY_DOUBLE);
    solution = (PyArrayObject *)PyArray_SimpleNew(1, ldims, NPY_DOUBLE);
    model = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (!conv || !ref || !no || !basis || !solution || !model)
        goto fail;
    int x = stamp->xss[stamp->sscnt], y = stamp->yss[stamp->sscnt];
    for (int p = 0; p < npix; p++)
    {
        ((float *)PyArray_DATA(conv))[p] = (float)stamp->vectors[0][p];
        ((float *)PyArray_DATA(ref))[p] = (float)stamp->krefArea[p];
        ((float *)PyArray_DATA(no))[p] =
            noise[(y - s->hwKSStamp + p / w) * s->nx + x - s->hwKSStamp + p % w];
    }
    for (int v = 0; v < nv; v++)
        memcpy((double *)PyArray_DATA(basis) + (size_t)v * npix, stamp->vectors[v],
               npix * sizeof(double));
    mat = matrix_new(nc + 1);
    sol = calloc(nc + 1, sizeof(double));
    if (!mat || !sol)
    {
        PyErr_NoMemory();
        goto fail;
    }
    for (int m = 1; m <= nc; m++)
    {
        sol[m] = stamp->scprod[m];
        for (int n = 1; n <= m; n++)
            mat[m][n] = mat[n][m] = stamp->mat[m][n];
    }
    double det;
    int status = ludcmp(mat, nc, s->indx, &det);
    if (status < 0)
    {
        PyErr_NoMemory();
        goto fail;
    }
    if (status == 0)
        lubksb(mat, nc, s->indx, sol);
    else
        for (int m = 1; m <= nc; m++)
            sol[m] = NAN;
    memcpy(PyArray_DATA(solution), sol + 1, nc * sizeof(double));
    for (int p = 0; p < npix; p++)
        for (int k = 0; k < s->nCompKer; k++)
            ((float *)PyArray_DATA(model))[p] += (float)(sol[k + 1] * stamp->vectors[k][p]);
    if (!owned_item(d, "substamp_id", PyLong_FromLong(id)))
        goto fail;
#define PUT(key, value)                                                                            \
    do                                                                                             \
    {                                                                                              \
        PyObject *v = (PyObject *)(value);                                                         \
        (value) = NULL;                                                                            \
        if (!owned_item(d, key, v))                                                                \
            goto fail;                                                                             \
    } while (0)
    if (dir[0] == 't')
    {
        PUT("image_cutout", ref);
        PUT("template_cutout", conv);
    }
    else
    {
        PUT("image_cutout", conv);
        PUT("template_cutout", ref);
    }
    PUT("noise_cutout", no);
    PUT("basis_vectors", basis);
    PUT("local_solution", solution);
    PUT("convolved_model_local", model);
#undef PUT
    matrix_free(mat, nc + 1);
    free(sol);
    return d;
fail:
    Py_XDECREF(conv);
    Py_XDECREF(ref);
    Py_XDECREF(no);
    Py_XDECREF(basis);
    Py_XDECREF(solution);
    Py_XDECREF(model);
    Py_XDECREF(d);
    matrix_free(mat, nc + 1);
    free(sol);
    return NULL;
}
static PyObject *py_fit_stamps_and_get_fom(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *conv, *ref, *noise;
    PyObject *coords;
    const char *dir;
    if (!PyArg_ParseTuple(args, "O!O!O!O!sO!", &HotpantsState_Type, &obj, &PyArray_Type, &conv,
                          &PyArray_Type, &ref, &PyArray_Type, &noise, &dir, &PyList_Type, &coords))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s)
        return NULL;
    if (!image_ok(s, conv, NPY_FLOAT32, "convolved input") ||
        !image_ok(s, ref, NPY_FLOAT32, "reference") || !image_ok(s, noise, NPY_FLOAT32, "noise"))
        return NULL;
    if (strcmp(dir, "t") && strcmp(dir, "i"))
    {
        PyErr_SetString(PyExc_ValueError, "direction must be t or i");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(coords);
    int ng = 0;
    stamp_struct *stamps = NULL;
    PyObject *results = NULL;
    if (n > INT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "too many substamps");
        return NULL;
    }
    for (Py_ssize_t j = 0; j < n; j++)
    {
        int g, id, x, y;
        PyObject *d = PyList_GET_ITEM(coords, j);
        if (!dict_int(d, "stamp_group_id", &g) || !dict_int(d, "substamp_id", &id) ||
            !dict_int(d, "x", &x) || !dict_int(d, "y", &y))
            return NULL;
        if (g < 0 || g >= s->nStamps || id < 0)
        {
            PyErr_SetString(PyExc_ValueError, "invalid substamp/group id");
            return NULL;
        }
        if (!coords_ok(s, x, y))
            return NULL;
        if (g + 1 > ng)
            ng = g + 1;
    }
    results = PyList_New(0);
    if (!results)
        return NULL;
    if (!n)
        return Py_BuildValue("dN", (double)MAXVAL, results);
    stamps = calloc(ng, sizeof(*stamps));
    if (!stamps || allocateStamps(s, stamps, ng))
    {
        PyErr_NoMemory();
        goto fail;
    }
    for (Py_ssize_t j = 0; j < n; j++)
    {
        int g, x, y;
        PyObject *d = PyList_GET_ITEM(coords, j);
        if (!dict_int(d, "stamp_group_id", &g) || !dict_int(d, "x", &x) || !dict_int(d, "y", &y))
            goto fail;
        if (stamps[g].nss >= s->nKSStamps)
        {
            PyErr_SetString(PyExc_ValueError, "substamps exceed configured nss");
            goto fail;
        }
        int k = stamps[g].nss++;
        stamps[g].xss[k] = x;
        stamps[g].yss[k] = y;
    }
    getKernelVec(s);
    if (!native_ok(s))
        goto fail;
    for (Py_ssize_t j = 0; j < n; j++)
    {
        int g, id, x, y;
        PyObject *d = PyList_GET_ITEM(coords, j);
        if (!dict_int(d, "stamp_group_id", &g) || !dict_int(d, "substamp_id", &id) ||
            !dict_int(d, "x", &x) || !dict_int(d, "y", &y))
            goto fail;
        int k = 0;
        while (k < stamps[g].nss && (stamps[g].xss[k] != x || stamps[g].yss[k] != y))
            k++;
        stamps[g].sscnt = k;
        if (fillStamp(s, &stamps[g], PyArray_DATA(conv), PyArray_DATA(ref)) || !native_ok(s))
        {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "invalid stamp");
            goto fail;
        }
        PyObject *r = local_result(s, &stamps[g], PyArray_DATA(noise), dir, id);
        if (!r)
            goto fail;
        int ok = PyList_Append(results, r);
        Py_DECREF(r);
        if (ok < 0)
            goto fail;
    }
    /* Group selection and the subsequent global fit both start at substamp 0.
       Producing diagnostic cutouts above must not silently select the last one. */
    for (int g = 0; g < ng; g++)
    {
        if (!stamps[g].nss)
            continue;
        stamps[g].sscnt = 0;
        if (fillStamp(s, &stamps[g], PyArray_DATA(conv), PyArray_DATA(ref)) || !native_ok(s))
        {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "invalid stamp");
            goto fail;
        }
    }
    double fom = check_stamps(s, stamps, ng, PyArray_DATA(ref), PyArray_DATA(noise));
    if (!native_ok(s))
        goto fail;
    for (Py_ssize_t j = 0; j < n; j++)
    {
        int g;
        if (!dict_int(PyList_GET_ITEM(coords, j), "stamp_group_id", &g))
            goto fail;
        PyObject *r = PyList_GET_ITEM(results, j);
        if (!owned_item(r, "fom", PyFloat_FromDouble(stamps[g].diff)) ||
            !owned_item(r, "chi2", PyFloat_FromDouble(stamps[g].chi2)) ||
            !owned_item(
                r, "survived_check",
                PyBool_FromLong(isfinite(stamps[g].diff) && stamps[g].diff < s->kerSigReject)))
            goto fail;
    }
    freeStampMem(s, stamps, ng);
    free(stamps);
    return Py_BuildValue("dN", fom, results);
fail:
    freeStampMem(s, stamps, ng);
    free(stamps);
    Py_XDECREF(results);
    return NULL;
}
static PyObject *py_fit_kernel(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyObject *list;
    PyArrayObject *conv, *ref, *noise;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &HotpantsState_Type, &obj, &PyList_Type, &list,
                          &PyArray_Type, &conv, &PyArray_Type, &ref, &PyArray_Type, &noise))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s)
        return NULL;
    if (!image_ok(s, conv, NPY_FLOAT32, "convolved input") ||
        !image_ok(s, ref, NPY_FLOAT32, "reference") || !image_ok(s, noise, NPY_FLOAT32, "noise"))
        return NULL;
    Py_ssize_t n = PyList_GET_SIZE(list);
    if (n <= 0 || n > s->nStamps)
    {
        PyErr_SetString(PyExc_ValueError, "invalid number of stamp groups");
        return NULL;
    }
    stamp_struct *stamps = calloc(n, sizeof(*stamps));
    PyObject *coeffs = NULL, *stats = NULL, *survivors = NULL;
    if (!stamps || allocateStamps(s, stamps, (int)n))
    {
        PyErr_NoMemory();
        goto fail;
    }
    getKernelVec(s);
    if (!native_ok(s))
        goto fail;
    for (Py_ssize_t j = 0; j < n; j++)
    {
        if (!plain_keys(PyList_GET_ITEM(list, j)))
            goto fail;
        PyObject *d = PyList_GET_ITEM(list, j),
                 *subs = PyDict_Check(d) ? PyDict_GetItemString(d, "substamps") : NULL;
        if (!subs || !PyList_Check(subs) || PyList_GET_SIZE(subs) < 1 ||
            PyList_GET_SIZE(subs) > s->nKSStamps)
        {
            PyErr_SetString(PyExc_ValueError, "each group requires 1..nss substamps");
            goto fail;
        }
        stamps[j].nss = (int)PyList_GET_SIZE(subs);
        for (int k = 0; k < stamps[j].nss; k++)
        {
            int x, y;
            PyObject *xy = PyList_GET_ITEM(subs, k);
            if (!coordinate_tuple(xy, &x, &y))
            {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_ValueError, "substamp must be an (x, y) integer tuple");
                goto fail;
            }
            if (!coords_ok(s, x, y))
                goto fail;
            stamps[j].xss[k] = x;
            stamps[j].yss[k] = y;
        }
        if (fillStamp(s, &stamps[j], PyArray_DATA(conv), PyArray_DATA(ref)) || !native_ok(s))
        {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "invalid stamp");
            goto fail;
        }
    }
    npy_intp dims[] = {s->nCompTotal + 1};
    coeffs = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (!coeffs)
        goto fail;
    double mean = NAN, scatter = NAN;
    int skipped = 0;
    s->nS = (int)n;
    fitKernel(s, stamps, PyArray_DATA(ref), PyArray_DATA(conv), PyArray_DATA(noise),
              PyArray_DATA((PyArrayObject *)coeffs), &mean, &scatter, &skipped);
    if (!native_ok(s) || !kernel_ok(s, (PyArrayObject *)coeffs))
        goto fail;
    survivors = PyList_New(0);
    if (!survivors)
        goto fail;
    for (int j = 0; j < n; j++)
        if (stamps[j].sscnt < stamps[j].nss)
        {
            PyObject *idx = PyLong_FromLong(j);
            if (!idx)
                goto fail;
            int ok = PyList_Append(survivors, idx);
            Py_DECREF(idx);
            if (ok < 0)
                goto fail;
        }
    stats =
        Py_BuildValue("{s:d,s:d,s:i}", "meansig", mean, "scatter", scatter, "nskipped", skipped);
    if (!stats)
        goto fail;
    PyObject *active = PyList_New(n);
    if (!active)
        goto fail;
    for (int j = 0; j < n; j++)
    {
        PyObject *idx = PyLong_FromLong(stamps[j].sscnt < stamps[j].nss ? stamps[j].sscnt : -1);
        if (!idx)
        {
            Py_DECREF(active);
            goto fail;
        }
        PyList_SET_ITEM(active, j, idx);
    }
    if (!owned_item(stats, "active_substamp_indices", active))
        goto fail;
    freeStampMem(s, stamps, (int)n);
    free(stamps);
    return Py_BuildValue("NNN", coeffs, stats, survivors);
fail:
    freeStampMem(s, stamps, (int)n);
    free(stamps);
    Py_XDECREF(coeffs);
    Py_XDECREF(stats);
    Py_XDECREF(survivors);
    return NULL;
}
static PyObject *py_apply_kernel(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *a, *k, *noise;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &HotpantsState_Type, &obj, &PyArray_Type, &a,
                          &PyArray_Type, &k, &PyArray_Type, &noise))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s)
        return NULL;
    if (!image_ok(s, a, NPY_FLOAT32, "image") || !image_ok(s, noise, NPY_FLOAT32, "noise") ||
        !kernel_ok(s, k))
        return NULL;
    getKernelVec(s);
    if (!native_ok(s))
        return NULL;
    npy_intp dims[] = {s->ny, s->nx};
    size_t bytes = (size_t)s->nx * s->ny * sizeof(float);
    PyObject *conv = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0),
             *mask = PyArray_ZEROS(2, dims, NPY_INT32, 0), *variance = NULL;
    float *v = malloc(bytes);
    if (!conv || !mask || !v)
    {
        Py_XDECREF(conv);
        Py_XDECREF(mask);
        free(v);
        return PyErr_NoMemory();
    }
    memcpy(v, PyArray_DATA(noise), bytes);
    int *original = s->mRData;
    s->mRData = PyArray_DATA((PyArrayObject *)mask);
    spatial_convolve(s, PyArray_DATA(a), &v, s->nx, s->ny, PyArray_DATA(k),
                     PyArray_DATA((PyArrayObject *)conv), original);
    s->mRData = original;
    if (!native_ok(s))
    {
        free(v);
        Py_DECREF(conv);
        Py_DECREF(mask);
        return NULL;
    }
    variance = wrap_buffer(v, 2, dims, NPY_FLOAT32);
    if (!variance)
    {
        Py_DECREF(conv);
        Py_DECREF(mask);
        return NULL;
    }
    return Py_BuildValue("NNN", conv, mask, variance);
}
static PyObject *py_get_background_image(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *k;
    if (!PyArg_ParseTuple(args, "O!O!", &HotpantsState_Type, &obj, &PyArray_Type, &k))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s || !kernel_ok(s, k))
        return NULL;
    npy_intp dims[] = {s->ny, s->nx};
    PyObject *out = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!out)
        return NULL;
    float *v = PyArray_DATA((PyArrayObject *)out);
    for (int y = 0; y < s->ny; y++)
        for (int x = 0; x < s->nx; x++)
            v[x + s->nx * y] = (float)get_background(s, x, y, PyArray_DATA(k));
    return out;
}
static int stats_inputs(PyObject *args, hotpants_state_t **state, PyArrayObject **diff,
                        PyArrayObject **noise, PyArrayObject **mask)
{
    HotpantsStateObject *obj;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &HotpantsState_Type, &obj, &PyArray_Type, diff,
                          &PyArray_Type, noise, &PyArray_Type, mask))
        return 0;
    *state = begin(obj);
    if (!*state)
        return 0;
    return image_ok(*state, *diff, NPY_FLOAT32, "difference") &&
           image_ok(*state, *noise, NPY_FLOAT32, "noise") &&
           image_ok(*state, *mask, NPY_INT32, "mask");
}
static PyObject *py_rescale_noise_ok(PyObject *self, PyObject *args)
{
    hotpants_state_t *s;
    PyArrayObject *diff, *noise, *mask;
    if (!stats_inputs(args, &s, &diff, &noise, &mask))
        return NULL;
    hotpants_state_t local = *s;
    PyObject *mask_copy = PyArray_NewCopy(mask, NPY_CORDER);
    if (!mask_copy)
        return NULL;
    local.mRData = PyArray_DATA((PyArrayObject *)mask_copy);
    double sd = NAN, mean = NAN;
    int a = getStampStats3(&local, PyArray_DATA(diff), 0, 0, s->nx, s->ny, NULL, NULL, NULL, NULL,
                           &sd, NULL, NULL, 0, FLAG_OUTPUT_ISBAD, 5);
    int b = getStampStats3(&local, PyArray_DATA(noise), 0, 0, s->nx, s->ny, NULL, &mean, NULL, NULL,
                           NULL, NULL, NULL, 0, FLAG_OUTPUT_ISBAD, 5);
    Py_DECREF(mask_copy);
    if (!native_ok(&local))
        return NULL;
    if ((a && a != 3) || (b && b != 3) || !isfinite(sd) || !isfinite(mean) || mean <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "insufficient valid pixels for noise rescaling");
        return NULL;
    }
    PyObject *out = PyArray_NewCopy(noise, NPY_CORDER);
    if (!out)
        return NULL;
    double ratio = sd / mean;
    float *v = PyArray_DATA((PyArrayObject *)out);
    int *m = PyArray_DATA(mask);
    if (ratio > 1)
        for (int j = 0; j < s->nx * s->ny; j++)
            if ((m[j] & 0xff) && !(m[j] & FLAG_OUTPUT_ISBAD))
                v[j] *= (float)ratio;
    return out;
}
static PyObject *py_calculate_final_stats(PyObject *self, PyObject *args)
{
    hotpants_state_t *s;
    PyArrayObject *diff, *noise, *mask;
    if (!stats_inputs(args, &s, &diff, &noise, &mask))
        return NULL;
    hotpants_state_t local = *s;
    PyObject *mask_copy = PyArray_NewCopy(mask, NPY_CORDER);
    if (!mask_copy)
        return NULL;
    local.mRData = PyArray_DATA((PyArrayObject *)mask_copy);
    double mean = NAN, sd = NAN, nmean = NAN, x2 = NAN;
    int nx2 = 0;
    getStampStats3(&local, PyArray_DATA(diff), 0, 0, s->nx, s->ny, NULL, &mean, NULL, NULL, &sd,
                   NULL, NULL, 0, 0xffff, 5);
    getStampStats3(&local, PyArray_DATA(noise), 0, 0, s->nx, s->ny, NULL, &nmean, NULL, NULL, NULL,
                   NULL, NULL, 0, 0xffff, 5);
    getNoiseStats3(&local, PyArray_DATA(diff), PyArray_DATA(noise), &x2, &nx2, 0, 0xffff);
    Py_DECREF(mask_copy);
    if (!native_ok(&local))
        return NULL;
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:i}", "diff_mean", mean, "diff_std", sd, "noise_mean",
                         nmean, "x2norm", x2, "nx2norm", nx2);
}
static PyObject *py_visualize_kernel(PyObject *self, PyObject *args)
{
    HotpantsStateObject *obj;
    PyArrayObject *k;
    PyObject *xy;
    double factor;
    if (!PyArg_ParseTuple(args, "O!O!O!d", &HotpantsState_Type, &obj, &PyTuple_Type, &xy,
                          &PyArray_Type, &k, &factor))
        return NULL;
    hotpants_state_t *s = begin(obj);
    if (!s || !kernel_ok(s, k))
        return NULL;
    int x, y;
    if (!coordinate_tuple(xy, &x, &y))
        return NULL;
    if (x < 0 || y < 0 || x >= s->nx || y >= s->ny || !isfinite(factor) || factor < 1 ||
        factor * s->fwKernel > 46340)
    {
        PyErr_SetString(PyExc_ValueError, "invalid kernel position or size_factor (must be >= 1)");
        return NULL;
    }
    getKernelVec(s);
    if (!native_ok(s))
        return NULL;
    /* Evaluate the kernel at the requested science-image coordinate, not at
       the origin of a synthetic miniature image. */
    make_kernel(s, x, y, PyArray_DATA(k));
    int size = (int)(factor * s->fwKernel);
    npy_intp dims[] = {size, size};
    PyObject *out = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (!out)
        return NULL;
    int offset = size / 2 - s->hwKernel;
    float *v = PyArray_DATA((PyArrayObject *)out);
    for (int j = 0; j < s->fwKernel; j++)
        for (int i = 0; i < s->fwKernel; i++)
            v[(j + offset) * size + i + offset] = (float)s->kernel[j * s->fwKernel + i];
    return out;
}
static PyTypeObject HotpantsState_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "hotpants.hotpants_ext.HotpantsState",
    .tp_basicsize = sizeof(HotpantsStateObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = state_new,
    .tp_init = (initproc)state_init,
    .tp_dealloc = (destructor)state_dealloc,
};
static PyMethodDef methods[] = {
    {"make_input_mask", py_make_input_mask, METH_VARARGS, "Build input mask."},
    {"calculate_noise_image", py_calculate_noise_image, METH_VARARGS, "Build variance image."},
    {"find_stamps", py_find_stamps, METH_VARARGS, "Find fitting stamps."},
    {"fit_stamps_and_get_fom", py_fit_stamps_and_get_fom, METH_VARARGS, "Fit local stamps."},
    {"fit_kernel", py_fit_kernel, METH_VARARGS, "Fit global kernel."},
    {"apply_kernel", py_apply_kernel, METH_VARARGS, "Convolve an image and variance."},
    {"get_background_image", py_get_background_image, METH_VARARGS, "Evaluate background."},
    {"rescale_noise_ok", py_rescale_noise_ok, METH_VARARGS, "Rescale noise."},
    {"calculate_final_stats", py_calculate_final_stats, METH_VARARGS, "Measure image statistics."},
    {"visualize_kernel", py_visualize_kernel, METH_VARARGS, "Evaluate a local kernel."},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "hotpants_ext", NULL, -1, methods};
PyMODINIT_FUNC PyInit_hotpants_ext(void)
{
    import_array();
    if (PyType_Ready(&HotpantsState_Type) < 0)
        return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m)
        return NULL;
    Py_INCREF(&HotpantsState_Type);
    if (PyModule_AddObject(m, "HotpantsState", (PyObject *)&HotpantsState_Type) < 0)
    {
        Py_DECREF(&HotpantsState_Type);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
