#define mpn_toomMN_mul mpn_toom42_mul
#define mpn_toomMN_mul_itch mpn_toom42_mul_itch

#define MIN_AN 10
#define MIN_BN(an) (((an) + 7) >> 2)
#define MAX_BN(an) ((2*(an)-5) / (size_t) 3)

#include "toom-shared.h"
