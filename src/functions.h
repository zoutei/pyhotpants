/* Alard.c */
void        getKernelVec(hotpants_state_t *);
int         fillStamp(hotpants_state_t *, stamp_struct *, float *, float *);
double      *kernel_vector(hotpants_state_t *, int, int, int, int, int *);
double      *kernel_vector_PCA(hotpants_state_t *, int, int, int, int, int *);
void        xy_conv_stamp(hotpants_state_t *, stamp_struct *, float *, int, int);
void        xy_conv_stamp_PCA(hotpants_state_t *, stamp_struct *, float *, int, int);
void        fitKernel(hotpants_state_t *, stamp_struct *, float *, float *, float *, double *, double *, double *, int *);
void        build_matrix0(hotpants_state_t *, stamp_struct *);
void        build_scprod0(hotpants_state_t *, stamp_struct *, float *);
double      check_stamps(hotpants_state_t *, stamp_struct *, int, float *, float *);
void        build_matrix(hotpants_state_t *, stamp_struct *, int, double **);
void        build_scprod(hotpants_state_t *, stamp_struct *, int, float *, double *);
void        getStampSig(hotpants_state_t *, stamp_struct *, double *, float *, double *, double *, double *);
void        getFinalStampSig(hotpants_state_t *, stamp_struct *, float *, float *, double *);
char        check_again(hotpants_state_t *, stamp_struct *, double *, float *, float *, float *, double *, double *, int *);
void        spatial_convolve(hotpants_state_t *, float *, float **, int, int, double *, float *, int *);
double      make_kernel(hotpants_state_t *, int, int, double *);
double      get_background(hotpants_state_t *, int, int, double *);
void        make_model(hotpants_state_t *, stamp_struct *, double *, float *);
int         ludcmp(double **, int, int *, double *);
void        lubksb(double **, int, int *, double *);

/* Functions.c */
int         allocateStamps(hotpants_state_t *, stamp_struct *, int);
void        buildStamps(hotpants_state_t *, int, int, int, int, int *, int *, int, int, int,
                        stamp_struct *, stamp_struct *, float *, float *,
                        float, float);
void        cutStamp(float *, float *, int, int, int, int, int, stamp_struct *);
void        buildSigMask(stamp_struct *, int, int, int *);
int         cutSStamp(hotpants_state_t *, stamp_struct *, float *);
double      checkPsfCenter(hotpants_state_t *, float *, int, int, int, int, int, int, double, float, float,
			   int, int, int, int);
int         getPsfCenters(hotpants_state_t *, stamp_struct *, float *, int, int, double, int, int);
int         getPsfCentersORIG(hotpants_state_t *, stamp_struct *, float *, int, int, double, int, int);
int         getStampStats3(hotpants_state_t *, float *, int, int, int, int, double *, double *, double *, double *, double *, double *, double *, int, int, int);
void        getNoiseStats3(hotpants_state_t *, float *, float *, double *, int *, int, int);
int         stampStats(double *, int *, long, double *, double *, double *, double *, double *, double *, double *);
int         sigma_clip(hotpants_state_t *, float *, int, double *, double *, int);
float      *calculateAvgNoise(hotpants_state_t *, float *, int *, int, int, int, int, int);
void        freeStampMem(hotpants_state_t *state, stamp_struct *, int);
float       *makeNoiseImage4(hotpants_state_t *, float *, float, float);
void        getKernelInfo(char *);
void        readKernel(char *, int, double **, double **, int *, int *, int *, int *, double *, double *, double *, double *, double *, int *);
void        spreadMask(hotpants_state_t *, int *, int);
void        makeInputMask(hotpants_state_t *, float *, float *, int *);
void        makeOutputMask(float *, float, float, float *, float, float, int *, int *, int *);
void        fset(float *, double, int, int);
void        dfset(double *, double, int, int);
void        printError(int);
double      ran1(int *);
void        quick_sort(double *,int *, int);
int         imin(int, int);
int         imax(int, int);

/* armin */
void savexy(hotpants_state_t *, stamp_struct *, int, long, long, int);
void loadxyfile(hotpants_state_t *, char *,int);

/* mysinc.c */
int    swarp_remap(float *, float *, double, double, int, int,
		   int, float *, float *, int);
double luptonD(int, double);
double luptonD_appx(int, double);
void   make_lupton_kernel(double, double *, int);
void   lanczos3(double, double *);
void   lanczos4(double, double *);
void   lanczos(double, double *, int);
