#define mpn_toomMN_mul mpn_toom33_mul
#define mpn_toomMN_mul_itch mpn_toom33_mul_itch

/* Smaller sizes not supported; may lead to recursive calls to
   toom22_mul with invalid input size. */
#define MIN_AN MUL_TOOM33_THRESHOLD
#define MIN_BN(an) (1 + 2*(((an)+2)/(size_t) 3))

#define COUNT 1000

#include "toom-shared.h"
