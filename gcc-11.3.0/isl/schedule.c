/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* This program takes an isl_schedule_constraints object as input and
 * prints a schedule that satisfies those constraints.
 */

#include <isl/options.h>
#include <isl/schedule.h>

int main(int argc, char **argv)
{
	isl_ctx *ctx;
	isl_printer *p;
	isl_schedule_constraints *sc;
	isl_schedule *schedule;
	struct isl_options *options;

	options = isl_options_new_with_defaults();
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);
	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	sc = isl_schedule_constraints_read_from_file(ctx, stdin);
	schedule = isl_schedule_constraints_compute_schedule(sc);

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
	p = isl_printer_print_schedule(p, schedule);
	isl_printer_free(p);

	isl_schedule_free(schedule);

	isl_ctx_free(ctx);

	return 0;
}
