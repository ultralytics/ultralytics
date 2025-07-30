#ifndef ISL_BAND_PRIVATE_H
#define ISL_BAND_PRIVATE_H

#include <isl/aff.h>
#include <isl/band.h>
#include <isl/list.h>
#include <isl/schedule.h>

/* Information about a band within a schedule.
 *
 * n is the number of scheduling dimensions within the band.
 * coincident is an array of length n, indicating whether a scheduling dimension
 *	satisfies the coincidence constraints in the sense that
 *	the corresponding dependence distances are zero.
 * pma is the partial schedule corresponding to this band.
 * schedule is the schedule that contains this band.
 * parent is the parent of this band (or NULL if the band is a root).
 * children are the children of this band (or NULL if the band is a leaf).
 *
 * To avoid circular dependences in the reference counting,
 * the schedule and parent pointers are not reference counted.
 * isl_band_copy increments the reference count of schedule to ensure
 * that outside references to the band keep the schedule alive.
 */
struct isl_band {
	int ref;

	int n;
	int *coincident;

	isl_union_pw_multi_aff *pma;
	isl_schedule *schedule;
	isl_band *parent;
	isl_band_list *children;
};

#undef EL
#define EL isl_band

#include <isl_list_templ.h>

__isl_give isl_band *isl_band_alloc(isl_ctx *ctx);

__isl_give isl_union_map *isl_band_list_get_suffix_schedule(
	__isl_keep isl_band_list *list);

#endif
