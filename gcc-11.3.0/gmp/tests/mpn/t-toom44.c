#define mpn_toomMN_mul mpn_toom44_mul
#define mpn_toomMN_mul_itch mpn_toom44_mul_itch

/* Smaller sizes not supported; may lead to recursive calls to
   toom22_mul or toom33_mul with invalid input size. */
#define MIN_AN MUL_TOOM44_THRESHOLD
#define MIN_BN(an) (1 + 3*(((an)+3)>>2))

#define COUNT 1000

#include "toom-shared.h"
