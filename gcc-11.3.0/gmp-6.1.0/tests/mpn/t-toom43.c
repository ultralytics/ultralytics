#define mpn_toomMN_mul mpn_toom43_mul
#define mpn_toomMN_mul_itch mpn_toom43_mul_itch

#define MIN_AN 25
#define MIN_BN(an) (1 + 2*(((an)+3) >> 2))
#define MAX_BN(an) ((an)-3)

#include "toom-shared.h"
