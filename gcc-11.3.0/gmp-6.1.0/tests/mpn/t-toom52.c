#define mpn_toomMN_mul mpn_toom52_mul
#define mpn_toomMN_mul_itch mpn_toom52_mul_itch

#define MIN_AN 32
#define MIN_BN(an) (((an) + 9) / (size_t) 5)
#define MAX_BN(an) (((an) - 3) >> 1)

#include "toom-shared.h"
