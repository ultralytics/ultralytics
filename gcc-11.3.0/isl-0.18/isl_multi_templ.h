#include <isl/space.h>

#include <isl_multi_macro.h>

struct MULTI(BASE) {
	int ref;
	isl_space *space;

	int n;
	EL *p[1];
};

__isl_give MULTI(BASE) *CAT(MULTI(BASE),_alloc)(__isl_take isl_space *space);
