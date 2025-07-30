#define mpn_toomMN_mul mpn_toom6h_mul
#define mpn_toomMN_mul_itch mpn_toom6h_mul_itch

#define SIZE_LOG 11

/* Smaller sizes not supported; may lead to recursive calls to
   toom22_mul, toom33_mul, or toom44_mul with invalid input size. */
#define MIN_AN MUL_TOOM6H_MIN
#define MIN_BN(an) (MAX ((an*3)>>3, 46))

#define COUNT 1000

#include "toom-shared.h"
