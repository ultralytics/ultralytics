#ifndef ISL_SCHEDULE_CONSTRAINTS_H
#define ISL_SCHEDULE_CONSTRAINTS_H

#include <isl/schedule.h>

enum isl_edge_type {
	isl_edge_validity = 0,
	isl_edge_first = isl_edge_validity,
	isl_edge_coincidence,
	isl_edge_condition,
	isl_edge_conditional_validity,
	isl_edge_proximity,
	isl_edge_last = isl_edge_proximity,
	isl_edge_local
};

__isl_give isl_schedule_constraints *
isl_schedule_constraints_align_params(__isl_take isl_schedule_constraints *sc);

__isl_give isl_union_map *isl_schedule_constraints_get(
	__isl_keep isl_schedule_constraints *sc, enum isl_edge_type type);
__isl_give isl_schedule_constraints *isl_schedule_constraints_add(
	__isl_take isl_schedule_constraints *sc, enum isl_edge_type type,
	__isl_take isl_union_map *c);

int isl_schedule_constraints_n_basic_map(
	__isl_keep isl_schedule_constraints *sc);
int isl_schedule_constraints_n_map(__isl_keep isl_schedule_constraints *sc);

#endif
