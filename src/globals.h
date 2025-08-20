typedef struct
{
   int       x0,y0;       /* origin of stamp in region coords*/
   int       x,y;         /* center of stamp in region coords*/
   int       nx,ny;       /* size of stamp */
   int       *xss;        /* x location of test substamp centers */
   int       *yss;        /* y location of test substamp centers */
   int       nss;         /* number of detected substamps, 1 .. nss     */
   int       sscnt;       /* represents which nss to use,  0 .. nss - 1 */
   double    **vectors;   /* contains convolved image data */
   double    *krefArea;   /* contains kernel substamp data */
   double    **mat;       /* fitting matrices */
   double    *scprod;     /* kernel sum solution */
   double    sum;         /* sum of fabs, for sigma use in check_stamps */
   double    mean;
   double    median;
   double    mode;        /* sky estimate */
   double    sd;
   double    fwhm;
   double    lfwhm;
   double    chi2;        /* residual in kernel fitting */
   double    norm;        /* kernel sum */
   double    diff;        /* (norm - mean_ksum) * sqrt(sum) */
} stamp_struct;

#define MAXDIM        4
#define SCRLEN        256
#define MAXVAL        1e10
#define ZEROVAL       1e-10

/* FLAGS */
#define FLAG_BAD_PIXVAL         0x01   /* 1 */
#define FLAG_SAT_PIXEL          0x02   /* 2 */
#define FLAG_LOW_PIXEL          0x04   /* 4 */
#define FLAG_ISNAN              0x08   /* 8 */
#define FLAG_BAD_CONV           0x10   /* 16  */
#define FLAG_INPUT_MASK         0x20   /* 32  */
#define FLAG_OK_CONV            0x40   /* 64  */
#define FLAG_INPUT_ISBAD        0x80   /* 128 */
#define FLAG_T_BAD              0x100  /* 256  */
#define FLAG_T_SKIP             0x200  /* 512  */
#define FLAG_I_BAD              0x400  /* 1024 */
#define FLAG_I_SKIP             0x800  /* 2048 */
#define FLAG_OUTPUT_ISBAD       0x8000 /* 32768 */

/* A single struct to hold all state, replacing global variables */
typedef struct {
    int nx, ny;
    float tUThresh, tLThresh, tUKThresh, tGain, tRdnoise, tPedestal;
    float iUThresh, iLThresh, iUKThresh, iGain, iRdnoise, iPedestal;
    int hwKernel, kerOrder, bgOrder;
    float kerFitThresh, kerSigReject, kerFracMask;
    int nKSStamps, hwKSStamp;
    int nRegX, nRegY, nStampX, nStampY;
    char forceConvolve_str[2], photNormalize_str[2], figMerit_str[2];
    float fillVal, fillValNoise;
    int verbose;
    int ngauss;
    int *deg_fixe;
    float *sigma_gauss;
    int rPixX, rPixY;
    int fwKernel, fwStamp, hwStamp, fwKSStamp;
    int nStamps, nS, nCompKer, nComp, nBGVectors, nCompTotal;
    int nC;
    int cmpFile;
    int rescaleOK;
    int convolveVariance;
    int usePCA;
    float **PCA;
    
    // Pointers to temporary C arrays that need to be managed
    int *mRData;
    float *temp, *temp2;
    double *check_stack, *filter_x, *filter_y, **kernel_vec;
    double **wxy, *kernel_coeffs, *kernel, **check_mat, *check_vec;
    stamp_struct *tStamps;
    stamp_struct *iStamps;
    int *indx;

    char *template, *image, *outim;
    char *tNoiseIm, *iNoiseIm, *tMaskIm, *iMaskIm, *kernelImIn, *kernelImOut, *outMask;
    char *regFile;
    char *regKeyWord;
    int numRegKeyWord;
    int useFullSS;
    char *sstampFile;
    int findSSC;
    float statSig, kfSpreadMask1, kfSpreadMask2;
    int sameConv, doSum, inclNoiseImage, inclSigmaImage, inclConvImage, noClobber;
    int doKerInfo, outShort, outNShort;
    float outBzero, outBscale, outNiBzero, outNiBscale;
    int kcStep;
    int dummy;
    int savexyflag;
    char xyfilename[1000];
    float *xcmp,*ycmp;
    int Ncmp;
} hotpants_state_t;

