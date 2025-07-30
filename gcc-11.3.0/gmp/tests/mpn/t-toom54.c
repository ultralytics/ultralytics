#define mpn_toomMN_mul mpn_toom54_mul
#define mpn_toomMN_mul_itch mpn_toom54_mul_itch

#define MIN_AN 31
#define MIN_BN(an) ((3*(an) + 32) / (size_t) 5)		/* 3/5 */
#define MAX_BN(an) ((an) - 6)	                        /* 1/1 */

#include "toom-shared.h"
