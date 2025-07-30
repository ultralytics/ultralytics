#include <isl/space.h>
#include <isl/printer.h>

/* Internal data structure for isl_print_space.
 *
 * latex is set if that is the output format.
 * print_dim (if not NULL) is called on each dimension.
 * user is set by the caller of print_space and may be used inside print_dim.
 *
 * space is the global space that is being printed.  This field is set by
 *	print_space.
 * type is the tuple of the global space that is currently being printed.
 *	This field is set by print_space.
 */
struct isl_print_space_data {
	int latex;
	__isl_give isl_printer *(*print_dim)(__isl_take isl_printer *p,
		struct isl_print_space_data *data, unsigned pos);
	void *user;

	isl_space *space;
	enum isl_dim_type type;
};

__isl_give isl_printer *isl_print_space(__isl_keep isl_space *space,
	__isl_take isl_printer *p, int rational,
	struct isl_print_space_data *data);
