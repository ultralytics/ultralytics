#ifndef ISL_OPTIONS_PRIVATE_H
#define ISL_OPTIONS_PRIVATE_H

#include <isl/options.h>

struct isl_options {
	#define			ISL_CONTEXT_GBR		0
	#define			ISL_CONTEXT_LEXMIN	1
	unsigned		context;

	#define			ISL_GBR_NEVER	0
	#define			ISL_GBR_ONCE	1
	#define			ISL_GBR_ALWAYS	2
	unsigned		gbr;
	unsigned		gbr_only_first;

	#define			ISL_CLOSURE_ISL		0
	#define			ISL_CLOSURE_BOX		1
	unsigned		closure;

	int			bound;
	unsigned		on_error;

	#define			ISL_BERNSTEIN_FACTORS	1
	#define			ISL_BERNSTEIN_INTERVALS	2
	int			bernstein_recurse;

	int			bernstein_triangulate;

	int			pip_symmetry;

	#define			ISL_CONVEX_HULL_WRAP	0
	#define			ISL_CONVEX_HULL_FM	1
	int			convex;

	int			coalesce_bounded_wrapping;

	int			schedule_max_coefficient;
	int			schedule_max_constant_term;
	int			schedule_parametric;
	int			schedule_outer_coincidence;
	int			schedule_maximize_band_depth;
	int			schedule_maximize_coincidence;
	int			schedule_split_scaled;
	int			schedule_treat_coalescing;
	int			schedule_separate_components;
	int			schedule_whole_component;
	unsigned		schedule_algorithm;
	int			schedule_serialize_sccs;

	int			tile_scale_tile_loops;
	int			tile_shift_point_loops;

	char			*ast_iterator_type;
	int			ast_always_print_block;
	int			ast_print_macro_once;

	int			ast_build_atomic_upper_bound;
	int			ast_build_prefer_pdiv;
	int			ast_build_detect_min_max;
	int			ast_build_exploit_nested_bounds;
	int			ast_build_group_coscheduled;
	int			ast_build_separation_bounds;
	int			ast_build_scale_strides;
	int			ast_build_allow_else;
	int			ast_build_allow_or;

	int			print_stats;
	unsigned long		max_operations;
};

#endif
