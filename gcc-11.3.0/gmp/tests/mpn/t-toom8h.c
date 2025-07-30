#define mpn_toomMN_mul mpn_toom8h_mul
#define mpn_toomMN_mul_itch mpn_toom8h_mul_itch

#define SIZE_LOG 11

/* Smaller sizes not supported; may lead to recursive calls to
   toom{22,33,44,6h}_mul with invalid input size. */
#define MIN_AN MUL_TOOM8H_MIN

#define MIN_BN(an)			 \
(MAX(GMP_NUMB_BITS <= 10*3 ? (an*6)/10 : \
     GMP_NUMB_BITS <= 11*3 ? (an*5)/11 : \
     GMP_NUMB_BITS <= 12*3 ? (an*4)/12 : \
     (an*4)/13, 86) )

#define COUNT 1000

#include "toom-shared.h"
