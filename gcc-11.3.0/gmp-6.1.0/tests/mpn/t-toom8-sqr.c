#define mpn_toomN_sqr mpn_toom8_sqr
#define mpn_toomN_sqr_itch mpn_toom8_sqr_itch
#define MIN_AN MAX(SQR_TOOM3_THRESHOLD,MAX(SQR_TOOM4_THRESHOLD,MAX(SQR_TOOM6_THRESHOLD,MAX(SQR_TOOM8_THRESHOLD,MPN_TOOM8_SQR_MINSIZE))))
#define MAX_AN SQR_FFT_THRESHOLD

#define COUNT 250

#include "toom-sqr-shared.h"
