#define mpn_toomMN_mul mpn_toom62_mul
#define mpn_toomMN_mul_itch mpn_toom62_mul_itch

#define MIN_AN 31
#define MIN_BN(an) (((an) + 11) / (size_t) 6)
#define MAX_BN(an) ((2*(an) - 7) / (size_t) 5)

#include "toom-shared.h"
