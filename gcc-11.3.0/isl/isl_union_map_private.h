#define isl_union_set_list	isl_union_map_list
#define isl_union_set	isl_union_map
#include <isl/union_map.h>
#include <isl/union_set.h>

struct isl_union_map {
	int ref;
	isl_space *dim;

	struct isl_hash_table	table;
};

__isl_give isl_union_map *isl_union_map_reset_range_space(
	__isl_take isl_union_map *umap, __isl_take isl_space *space);
