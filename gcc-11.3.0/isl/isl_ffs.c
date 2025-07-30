#include <isl_config.h>

#if !HAVE_DECL_FFS && !HAVE_DECL___BUILTIN_FFS && HAVE_DECL__BITSCANFORWARD
#include <intrin.h>

/* Implementation of ffs in terms of _BitScanForward.
 *
 * ffs returns the position of the least significant bit set in i,
 * with the least significant bit is position 1, or 0 if not bits are set.
 *
 * _BitScanForward returns 1 if mask is non-zero and sets index
 * to the position of the least significant bit set in i,
 * with the least significant bit is position 0.
 */
int isl_ffs(int i)
{
	unsigned char non_zero;
	unsigned long index, mask = i;

	non_zero = _BitScanForward(&index, mask);

	return non_zero ? 1 + index : 0;
}
#endif
