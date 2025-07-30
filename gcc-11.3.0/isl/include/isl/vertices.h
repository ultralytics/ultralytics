#ifndef ISL_VERTICES_H
#define ISL_VERTICES_H

#include <isl/aff_type.h>
#include <isl/set_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_external_vertex;
typedef struct isl_external_vertex	isl_vertex;

struct isl_cell;
typedef struct isl_cell		isl_cell;

struct isl_vertices;
typedef struct isl_vertices	isl_vertices;

isl_ctx *isl_vertex_get_ctx(__isl_keep isl_vertex *vertex);
int isl_vertex_get_id(__isl_keep isl_vertex *vertex);
__isl_give isl_basic_set *isl_vertex_get_domain(__isl_keep isl_vertex *vertex);
__isl_give isl_multi_aff *isl_vertex_get_expr(__isl_keep isl_vertex *vertex);
void isl_vertex_free(__isl_take isl_vertex *vertex);

__isl_give isl_vertices *isl_basic_set_compute_vertices(
	__isl_keep isl_basic_set *bset);
isl_ctx *isl_vertices_get_ctx(__isl_keep isl_vertices *vertices);
int isl_vertices_get_n_vertices(__isl_keep isl_vertices *vertices);
isl_stat isl_vertices_foreach_vertex(__isl_keep isl_vertices *vertices,
	isl_stat (*fn)(__isl_take isl_vertex *vertex, void *user), void *user);
void isl_vertices_free(__isl_take isl_vertices *vertices);

isl_ctx *isl_cell_get_ctx(__isl_keep isl_cell *cell);
__isl_give isl_basic_set *isl_cell_get_domain(__isl_keep isl_cell *cell);
isl_stat isl_cell_foreach_vertex(__isl_keep isl_cell *cell,
	isl_stat (*fn)(__isl_take isl_vertex *vertex, void *user), void *user);
void isl_cell_free(__isl_take isl_cell *cell);

isl_stat isl_vertices_foreach_cell(__isl_keep isl_vertices *vertices,
	isl_stat (*fn)(__isl_take isl_cell *cell, void *user), void *user);

#if defined(__cplusplus)
}
#endif

#endif
