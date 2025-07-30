#include "isl_config.h"
#include "gitversion.h"

const char *isl_version(void)
{
	return GIT_HEAD_ID
#ifdef USE_GMP_FOR_MP
	"-GMP"
#endif
#ifdef USE_IMATH_FOR_MP
	"-IMath"
#ifdef USE_SMALL_INT_OPT
	"-32"
#endif
#endif
	"\n";
}
