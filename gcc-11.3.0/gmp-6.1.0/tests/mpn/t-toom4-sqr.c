#define mpn_toomN_sqr mpn_toom4_sqr
#define mpn_toomN_sqr_itch mpn_toom4_sqr_itch
#define MIN_AN MAX(SQR_TOOM3_THRESHOLD,MAX(SQR_TOOM4_THRESHOLD,MPN_TOOM4_SQR_MINSIZE))
#define MAX_AN SQR_TOOM6_THRESHOLD

#include "toom-sqr-shared.h"
