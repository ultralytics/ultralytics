#include <assert.h>
#include <isl/map.h>
#include <isl/options.h>

int main(int argc, char **argv)
{
	struct isl_ctx *ctx;
	struct isl_map *map;
	struct isl_options *options;
	isl_printer *p;
	int exact;

	options = isl_options_new_with_defaults();
	assert(options);
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	p = isl_printer_to_file(ctx, stdout);

	map = isl_map_read_from_file(ctx, stdin);
	map = isl_map_transitive_closure(map, &exact);
	if (!exact)
		p = isl_printer_print_str(p, "# NOT exact\n");
	p = isl_printer_print_map(p, map);
	p = isl_printer_end_line(p);
	map = isl_map_compute_divs(map);
	map = isl_map_coalesce(map);
	p = isl_printer_print_str(p, "# coalesced\n");
	p = isl_printer_print_map(p, map);
	p = isl_printer_end_line(p);
	isl_map_free(map);

	isl_printer_free(p);

	isl_ctx_free(ctx);

	return 0;
}
