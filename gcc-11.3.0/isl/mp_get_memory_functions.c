#include <gmp.h>

void mp_get_memory_functions(
		void *(**alloc_func_ptr) (size_t),
		void *(**realloc_func_ptr) (void *, size_t, size_t),
		void (**free_func_ptr) (void *, size_t))
{
	if (alloc_func_ptr)
		*alloc_func_ptr = __gmp_allocate_func;
	if (realloc_func_ptr)
		*realloc_func_ptr = __gmp_reallocate_func;
	if (free_func_ptr)
		*free_func_ptr = __gmp_free_func;
}
