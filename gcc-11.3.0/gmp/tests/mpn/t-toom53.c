#define mpn_toomMN_mul mpn_toom53_mul
#define mpn_toomMN_mul_itch mpn_toom53_mul_itch

#define MIN_AN 17
#define MIN_BN(an) (1 + 2*(((an) + 4) / (size_t) 5))
#define MAX_BN(an) ((3*(an) - 11) >> 2)

#include "toom-shared.h"
