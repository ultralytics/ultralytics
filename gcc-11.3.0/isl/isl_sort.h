#ifndef ISL_SORT_H
#define ISL_SORT_H

#include <stddef.h>

int isl_sort(void *const pbase, size_t total_elems, size_t size,
	int (*cmp)(const void *, const void *, void *arg), void *arg);

#endif
