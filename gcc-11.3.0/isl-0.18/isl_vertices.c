/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <isl_map_private.h>
#include <isl_aff_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_tab.h>
#include <isl_space_private.h>
#include <isl_morph.h>
#include <isl_vertices_private.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>

#define SELECTED	1
#define DESELECTED	-1
#define UNSELECTED	0

static __isl_give isl_vertices *compute_chambers(__isl_take isl_basic_set *bset,
	__isl_take isl_vertices *vertices);

__isl_give isl_vertices *isl_vertices_copy(__isl_keep isl_vertices *vertices)
{
	if (!vertices)
		return NULL;

	vertices->ref++;
	return vertices;
}

void isl_vertices_free(__isl_take isl_vertices *vertices)
{
	int i;

	if (!vertices)
		return;

	if (--vertices->ref > 0)
		return;

	for (i = 0; i < vertices->n_vertices; ++i) {
		isl_basic_set_free(vertices->v[i].vertex);
		isl_basic_set_free(vertices->v[i].dom);
	}
	free(vertices->v);

	for (i = 0; i < vertices->n_chambers; ++i) {
		free(vertices->c[i].vertices);
		isl_basic_set_free(vertices->c[i].dom);
	}
	free(vertices->c);

	isl_basic_set_free(vertices->bset);
	free(vertices);
}

struct isl_vertex_list {
	struct isl_vertex v;
	struct isl_vertex_list *next;
};

static void free_vertex_list(struct isl_vertex_list *list)
{
	struct isl_vertex_list *next;

	for (; list; list = next) {
		next = list->next;
		isl_basic_set_free(list->v.vertex);
		isl_basic_set_free(list->v.dom);
		free(list);
	}
}

static __isl_give isl_vertices *vertices_from_list(__isl_keep isl_basic_set *bset,
	int n_vertices, struct isl_vertex_list *list)
{
	int i;
	struct isl_vertex_list *next;
	isl_vertices *vertices;

	vertices = isl_calloc_type(bset->ctx, isl_vertices);
	if (!vertices)
		goto error;
	vertices->ref = 1;
	vertices->bset = isl_basic_set_copy(bset);
	vertices->v = isl_alloc_array(bset->ctx, struct isl_vertex, n_vertices);
	if (n_vertices && !vertices->v)
		goto error;
	vertices->n_vertices = n_vertices;

	for (i = 0; list; list = next, i++) {
		next = list->next;
		vertices->v[i] = list->v;
		free(list);
	}

	return vertices;
error:
	isl_vertices_free(vertices);
	free_vertex_list(list);
	return NULL;
}

/* Prepend a vertex to the linked list "list" based on the equalities in "tab".
 * Return isl_bool_true if the vertex was actually added and
 * isl_bool_false otherwise.
 * In particular, vertices with a lower-dimensional activity domain are
 * not added to the list because they would not be included in any chamber.
 * Return isl_bool_error on error.
 */
static isl_bool add_vertex(struct isl_vertex_list **list,
	__isl_keep isl_basic_set *bset, struct isl_tab *tab)
{
	unsigned nvar;
	struct isl_vertex_list *v = NULL;

	if (isl_tab_detect_implicit_equalities(tab) < 0)
		return isl_bool_error;

	nvar = isl_basic_set_dim(bset, isl_dim_set);

	v = isl_calloc_type(tab->mat->ctx, struct isl_vertex_list);
	if (!v)
		goto error;

	v->v.vertex = isl_basic_set_copy(bset);
	v->v.vertex = isl_basic_set_cow(v->v.vertex);
	v->v.vertex = isl_basic_set_update_from_tab(v->v.vertex, tab);
	v->v.vertex = isl_basic_set_simplify(v->v.vertex);
	v->v.vertex = isl_basic_set_finalize(v->v.vertex);
	if (!v->v.vertex)
		goto error;
	isl_assert(bset->ctx, v->v.vertex->n_eq >= nvar, goto error);
	v->v.dom = isl_basic_set_copy(v->v.vertex);
	v->v.dom = isl_basic_set_params(v->v.dom);
	if (!v->v.dom)
		goto error;

	if (v->v.dom->n_eq > 0) {
		free_vertex_list(v);
		return isl_bool_false;
	}

	v->next = *list;
	*list = v;

	return isl_bool_true;
error:
	free_vertex_list(v);
	return isl_bool_error;
}

/* Compute the parametric vertices and the chamber decomposition
 * of an empty parametric polytope.
 */
static __isl_give isl_vertices *vertices_empty(__isl_keep isl_basic_set *bset)
{
	isl_vertices *vertices;

	if (!bset)
		return NULL;

	vertices = isl_calloc_type(bset->ctx, isl_vertices);
	if (!vertices)
		return NULL;
	vertices->bset = isl_basic_set_copy(bset);
	vertices->ref = 1;

	vertices->n_vertices = 0;
	vertices->n_chambers = 0;

	return vertices;
}

/* Compute the parametric vertices and the chamber decomposition
 * of the parametric polytope defined using the same constraints
 * as "bset" in the 0D case.
 * There is exactly one 0D vertex and a single chamber containing
 * the vertex.
 */
static __isl_give isl_vertices *vertices_0D(__isl_keep isl_basic_set *bset)
{
	isl_vertices *vertices;

	if (!bset)
		return NULL;

	vertices = isl_calloc_type(bset->ctx, isl_vertices);
	if (!vertices)
		return NULL;
	vertices->ref = 1;
	vertices->bset = isl_basic_set_copy(bset);

	vertices->v = isl_calloc_array(bset->ctx, struct isl_vertex, 1);
	if (!vertices->v)
		goto error;
	vertices->n_vertices = 1;
	vertices->v[0].vertex = isl_basic_set_copy(bset);
	vertices->v[0].dom = isl_basic_set_params(isl_basic_set_copy(bset));
	if (!vertices->v[0].vertex || !vertices->v[0].dom)
		goto error;

	vertices->c = isl_calloc_array(bset->ctx, struct isl_chamber, 1);
	if (!vertices->c)
		goto error;
	vertices->n_chambers = 1;
	vertices->c[0].n_vertices = 1;
	vertices->c[0].vertices = isl_calloc_array(bset->ctx, int, 1);
	if (!vertices->c[0].vertices)
		goto error;
	vertices->c[0].dom = isl_basic_set_copy(vertices->v[0].dom);
	if (!vertices->c[0].dom)
		goto error;

	return vertices;
error:
	isl_vertices_free(vertices);
	return NULL;
}

static int isl_mat_rank(__isl_keep isl_mat *mat)
{
	int row, col;
	isl_mat *H;

	H = isl_mat_left_hermite(isl_mat_copy(mat), 0, NULL, NULL);
	if (!H)
		return -1;

	for (col = 0; col < H->n_col; ++col) {
		for (row = 0; row < H->n_row; ++row)
			if (!isl_int_is_zero(H->row[row][col]))
				break;
		if (row == H->n_row)
			break;
	}

	isl_mat_free(H);

	return col;
}

/* Is the row pointed to by "f" linearly independent of the "n" first
 * rows in "facets"?
 */
static int is_independent(__isl_keep isl_mat *facets, int n, isl_int *f)
{
	int rank;

	if (isl_seq_first_non_zero(f, facets->n_col) < 0)
		return 0;

	isl_seq_cpy(facets->row[n], f, facets->n_col);
	facets->n_row = n + 1;
	rank = isl_mat_rank(facets);
	if (rank < 0)
		return -1;

	return rank == n + 1;
}

/* Check whether we can select constraint "level", given the current selection
 * reflected by facets in "tab", the rows of "facets" and the earlier
 * "selected" elements of "selection".
 *
 * If the constraint is (strictly) redundant in the tableau, selecting it would
 * result in an empty tableau, so it can't be selected.
 * If the set variable part of the constraint is not linearly independent
 * of the set variable parts of the already selected constraints,
 * the constraint cannot be selected.
 * If selecting the constraint results in an empty tableau, the constraint
 * cannot be selected.
 * Finally, if selecting the constraint results in some explicitly
 * deselected constraints turning into equalities, then the corresponding
 * vertices have already been generated, so the constraint cannot be selected.
 */
static int can_select(__isl_keep isl_basic_set *bset, int level,
	struct isl_tab *tab, __isl_keep isl_mat *facets, int selected,
	int *selection)
{
	int i;
	int indep;
	unsigned ovar;
	struct isl_tab_undo *snap;

	if (isl_tab_is_redundant(tab, level))
		return 0;

	ovar = isl_space_offset(bset->dim, isl_dim_set);

	indep = is_independent(facets, selected, bset->ineq[level] + 1 + ovar);
	if (indep < 0)
		return -1;
	if (!indep)
		return 0;

	snap = isl_tab_snap(tab);
	if (isl_tab_select_facet(tab, level) < 0)
		return -1;

	if (tab->empty) {
		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
		return 0;
	}

	for (i = 0; i < level; ++i) {
		int sgn;

		if (selection[i] != DESELECTED)
			continue;

		if (isl_tab_is_equality(tab, i))
			sgn = 0;
		else if (isl_tab_is_redundant(tab, i))
			sgn = 1;
		else
			sgn = isl_tab_sign_of_max(tab, i);
		if (sgn < -1)
			return -1;
		if (sgn <= 0) {
			if (isl_tab_rollback(tab, snap) < 0)
				return -1;
			return 0;
		}
	}

	return 1;
}

/* Compute the parametric vertices and the chamber decomposition
 * of a parametric polytope that is not full-dimensional.
 *
 * Simply map the parametric polytope to a lower dimensional space
 * and map the resulting vertices back.
 */
static __isl_give isl_vertices *lower_dim_vertices(
	__isl_keep isl_basic_set *bset)
{
	isl_morph *morph;
	isl_vertices *vertices;

	bset = isl_basic_set_copy(bset);
	morph = isl_basic_set_full_compression(bset);
	bset = isl_morph_basic_set(isl_morph_copy(morph), bset);

	vertices = isl_basic_set_compute_vertices(bset);
	isl_basic_set_free(bset);

	morph = isl_morph_inverse(morph);

	vertices = isl_morph_vertices(morph, vertices);

	return vertices;
}

/* Compute the parametric vertices and the chamber decomposition
 * of the parametric polytope defined using the same constraints
 * as "bset".  "bset" is assumed to have no existentially quantified
 * variables.
 *
 * The vertices themselves are computed in a fairly simplistic way.
 * We simply run through all combinations of d constraints,
 * with d the number of set variables, and check if those d constraints
 * define a vertex.  To avoid the generation of duplicate vertices,
 * which we may happen if a vertex is defined by more that d constraints,
 * we make sure we only generate the vertex for the d constraints with
 * smallest index.
 *
 * We set up a tableau and keep track of which facets have been
 * selected.  The tableau is marked strict_redundant so that we can be
 * sure that any constraint that is marked redundant (and that is not
 * also marked zero) is not an equality.
 * If a constraint is marked DESELECTED, it means the constraint was
 * SELECTED before (in combination with the same selection of earlier
 * constraints).  If such a deselected constraint turns out to be an
 * equality, then any vertex that may still be found with the current
 * selection has already been generated when the constraint was selected.
 * A constraint is marked UNSELECTED when there is no way selecting
 * the constraint could lead to a vertex (in combination with the current
 * selection of earlier constraints).
 *
 * The set variable coefficients of the selected constraints are stored
 * in the facets matrix.
 */
__isl_give isl_vertices *isl_basic_set_compute_vertices(
	__isl_keep isl_basic_set *bset)
{
	struct isl_tab *tab;
	int level;
	int init;
	unsigned nvar;
	int *selection = NULL;
	int selected;
	struct isl_tab_undo **snap = NULL;
	isl_mat *facets = NULL;
	struct isl_vertex_list *list = NULL;
	int n_vertices = 0;
	isl_vertices *vertices;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return vertices_empty(bset);

	if (bset->n_eq != 0)
		return lower_dim_vertices(bset);

	isl_assert(bset->ctx, isl_basic_set_dim(bset, isl_dim_div) == 0,
		return NULL);

	if (isl_basic_set_dim(bset, isl_dim_set) == 0)
		return vertices_0D(bset);

	nvar = isl_basic_set_dim(bset, isl_dim_set);

	bset = isl_basic_set_copy(bset);
	bset = isl_basic_set_set_rational(bset);
	if (!bset)
		return NULL;

	tab = isl_tab_from_basic_set(bset, 0);
	if (!tab)
		goto error;
	tab->strict_redundant = 1;

	if (tab->empty)	{
		vertices = vertices_empty(bset);
		isl_basic_set_free(bset);
		isl_tab_free(tab);
		return vertices;
	}

	selection = isl_alloc_array(bset->ctx, int, bset->n_ineq);
	snap = isl_alloc_array(bset->ctx, struct isl_tab_undo *, bset->n_ineq);
	facets = isl_mat_alloc(bset->ctx, nvar, nvar);
	if ((bset->n_ineq && (!selection || !snap)) || !facets)
		goto error;

	level = 0;
	init = 1;
	selected = 0;

	while (level >= 0) {
		if (level >= bset->n_ineq ||
		    (!init && selection[level] != SELECTED)) {
			--level;
			init = 0;
			continue;
		}
		if (init) {
			int ok;
			snap[level] = isl_tab_snap(tab);
			ok = can_select(bset, level, tab, facets, selected,
					selection);
			if (ok < 0)
				goto error;
			if (ok) {
				selection[level] = SELECTED;
				selected++;
			} else
				selection[level] = UNSELECTED;
		} else {
			selection[level] = DESELECTED;
			selected--;
			if (isl_tab_rollback(tab, snap[level]) < 0)
				goto error;
		}
		if (selected == nvar) {
			if (tab->n_dead == nvar) {
				isl_bool added = add_vertex(&list, bset, tab);
				if (added < 0)
					goto error;
				if (added)
					n_vertices++;
			}
			init = 0;
			continue;
		}
		++level;
		init = 1;
	}

	isl_mat_free(facets);
	free(selection);
	free(snap);

	isl_tab_free(tab);

	vertices = vertices_from_list(bset, n_vertices, list);

	vertices = compute_chambers(bset, vertices);

	return vertices;
error:
	free_vertex_list(list);
	isl_mat_free(facets);
	free(selection);
	free(snap);
	isl_tab_free(tab);
	isl_basic_set_free(bset);
	return NULL;
}

struct isl_chamber_list {
	struct isl_chamber c;
	struct isl_chamber_list *next;
};

static void free_chamber_list(struct isl_chamber_list *list)
{
	struct isl_chamber_list *next;

	for (; list; list = next) {
		next = list->next;
		isl_basic_set_free(list->c.dom);
		free(list->c.vertices);
		free(list);
	}
}

/* Check whether the basic set "bset" is a superset of the basic set described
 * by "tab", i.e., check whether all constraints of "bset" are redundant.
 */
static int bset_covers_tab(__isl_keep isl_basic_set *bset, struct isl_tab *tab)
{
	int i;

	if (!bset || !tab)
		return -1;

	for (i = 0; i < bset->n_ineq; ++i) {
		enum isl_ineq_type type = isl_tab_ineq_type(tab, bset->ineq[i]);
		switch (type) {
		case isl_ineq_error:		return -1;
		case isl_ineq_redundant:	continue;
		default:			return 0;
		}
	}

	return 1;
}

static __isl_give isl_vertices *vertices_add_chambers(
	__isl_take isl_vertices *vertices, int n_chambers,
	struct isl_chamber_list *list)
{
	int i;
	isl_ctx *ctx;
	struct isl_chamber_list *next;

	ctx = isl_vertices_get_ctx(vertices);
	vertices->c = isl_alloc_array(ctx, struct isl_chamber, n_chambers);
	if (!vertices->c)
		goto error;
	vertices->n_chambers = n_chambers;

	for (i = 0; list; list = next, i++) {
		next = list->next;
		vertices->c[i] = list->c;
		free(list);
	}

	return vertices;
error:
	isl_vertices_free(vertices);
	free_chamber_list(list);
	return NULL;
}

/* Can "tab" be intersected with "bset" without resulting in
 * a lower-dimensional set.
 * "bset" itself is assumed to be full-dimensional.
 */
static isl_bool can_intersect(struct isl_tab *tab,
	__isl_keep isl_basic_set *bset)
{
	int i;
	struct isl_tab_undo *snap;

	if (bset->n_eq > 0)
		isl_die(isl_basic_set_get_ctx(bset), isl_error_internal,
			"expecting full-dimensional input",
			return isl_bool_error);

	if (isl_tab_extend_cons(tab, bset->n_ineq) < 0)
		return isl_bool_error;

	snap = isl_tab_snap(tab);

	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_tab_ineq_type(tab, bset->ineq[i]) == isl_ineq_redundant)
			continue;
		if (isl_tab_add_ineq(tab, bset->ineq[i]) < 0)
			return isl_bool_error;
	}

	if (isl_tab_detect_implicit_equalities(tab) < 0)
		return isl_bool_error;
	if (tab->n_dead) {
		if (isl_tab_rollback(tab, snap) < 0)
			return isl_bool_error;
		return isl_bool_false;
	}

	return isl_bool_true;
}

static int add_chamber(struct isl_chamber_list **list,
	__isl_keep isl_vertices *vertices, struct isl_tab *tab, int *selection)
{
	int n_frozen;
	int i, j;
	int n_vertices = 0;
	struct isl_tab_undo *snap;
	struct isl_chamber_list *c = NULL;

	for (i = 0; i < vertices->n_vertices; ++i)
		if (selection[i])
			n_vertices++;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con && tab->con[i].frozen; ++i)
		tab->con[i].frozen = 0;
	n_frozen = i;

	if (isl_tab_detect_redundant(tab) < 0)
		return -1;

	c = isl_calloc_type(tab->mat->ctx, struct isl_chamber_list);
	if (!c)
		goto error;
	c->c.vertices = isl_alloc_array(tab->mat->ctx, int, n_vertices);
	if (n_vertices && !c->c.vertices)
		goto error;
	c->c.dom = isl_basic_set_copy(isl_tab_peek_bset(tab));
	c->c.dom = isl_basic_set_set_rational(c->c.dom);
	c->c.dom = isl_basic_set_cow(c->c.dom);
	c->c.dom = isl_basic_set_update_from_tab(c->c.dom, tab);
	c->c.dom = isl_basic_set_simplify(c->c.dom);
	c->c.dom = isl_basic_set_finalize(c->c.dom);
	if (!c->c.dom)
		goto error;

	c->c.n_vertices = n_vertices;

	for (i = 0, j = 0; i < vertices->n_vertices; ++i)
		if (selection[i]) {
			c->c.vertices[j] = i;
			j++;
		}

	c->next = *list;
	*list = c;

	for (i = 0; i < n_frozen; ++i)
		tab->con[i].frozen = 1;

	if (isl_tab_rollback(tab, snap) < 0)
		return -1;

	return 0;
error:
	free_chamber_list(c);
	return -1;
}

struct isl_facet_todo {
	struct isl_tab *tab;	/* A tableau representation of the facet */
	isl_basic_set *bset;    /* A normalized basic set representation */
	isl_vec *constraint;	/* Constraint pointing to the other side */
	struct isl_facet_todo *next;
};

static void free_todo(struct isl_facet_todo *todo)
{
	while (todo) {
		struct isl_facet_todo *next = todo->next;

		isl_tab_free(todo->tab);
		isl_basic_set_free(todo->bset);
		isl_vec_free(todo->constraint);
		free(todo);

		todo = next;
	}
}

static struct isl_facet_todo *create_todo(struct isl_tab *tab, int con)
{
	int i;
	int n_frozen;
	struct isl_tab_undo *snap;
	struct isl_facet_todo *todo;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con && tab->con[i].frozen; ++i)
		tab->con[i].frozen = 0;
	n_frozen = i;

	if (isl_tab_detect_redundant(tab) < 0)
		return NULL;

	todo = isl_calloc_type(tab->mat->ctx, struct isl_facet_todo);
	if (!todo)
		return NULL;

	todo->constraint = isl_vec_alloc(tab->mat->ctx, 1 + tab->n_var);
	if (!todo->constraint)
		goto error;
	isl_seq_neg(todo->constraint->el, tab->bmap->ineq[con], 1 + tab->n_var);
	todo->bset = isl_basic_set_copy(isl_tab_peek_bset(tab));
	todo->bset = isl_basic_set_set_rational(todo->bset);
	todo->bset = isl_basic_set_cow(todo->bset);
	todo->bset = isl_basic_set_update_from_tab(todo->bset, tab);
	todo->bset = isl_basic_set_simplify(todo->bset);
	todo->bset = isl_basic_set_sort_constraints(todo->bset);
	if (!todo->bset)
		goto error;
	ISL_F_SET(todo->bset, ISL_BASIC_SET_NORMALIZED);
	todo->tab = isl_tab_dup(tab);
	if (!todo->tab)
		goto error;

	for (i = 0; i < n_frozen; ++i)
		tab->con[i].frozen = 1;

	if (isl_tab_rollback(tab, snap) < 0)
		goto error;

	return todo;
error:
	free_todo(todo);
	return NULL;
}

/* Create todo items for all interior facets of the chamber represented
 * by "tab" and collect them in "next".
 */
static int init_todo(struct isl_facet_todo **next, struct isl_tab *tab)
{
	int i;
	struct isl_tab_undo *snap;
	struct isl_facet_todo *todo;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con; ++i) {
		if (tab->con[i].frozen)
			continue;
		if (tab->con[i].is_redundant)
			continue;

		if (isl_tab_select_facet(tab, i) < 0)
			return -1;

		todo = create_todo(tab, i);
		if (!todo)
			return -1;

		todo->next = *next;
		*next = todo;

		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
	}

	return 0;
}

/* Does the linked list contain a todo item that is the opposite of "todo".
 * If so, return 1 and remove the opposite todo item.
 */
static int has_opposite(struct isl_facet_todo *todo,
	struct isl_facet_todo **list)
{
	for (; *list; list = &(*list)->next) {
		int eq;
		eq = isl_basic_set_plain_is_equal(todo->bset, (*list)->bset);
		if (eq < 0)
			return -1;
		if (!eq)
			continue;
		todo = *list;
		*list = todo->next;
		todo->next = NULL;
		free_todo(todo);
		return 1;
	}

	return 0;
}

/* Create todo items for all interior facets of the chamber represented
 * by "tab" and collect them in first->next, taking care to cancel
 * opposite todo items.
 */
static int update_todo(struct isl_facet_todo *first, struct isl_tab *tab)
{
	int i;
	struct isl_tab_undo *snap;
	struct isl_facet_todo *todo;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con; ++i) {
		int drop;

		if (tab->con[i].frozen)
			continue;
		if (tab->con[i].is_redundant)
			continue;

		if (isl_tab_select_facet(tab, i) < 0)
			return -1;

		todo = create_todo(tab, i);
		if (!todo)
			return -1;

		drop = has_opposite(todo, &first->next);
		if (drop < 0)
			return -1;

		if (drop)
			free_todo(todo);
		else {
			todo->next = first->next;
			first->next = todo;
		}

		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
	}

	return 0;
}

/* Compute the chamber decomposition of the parametric polytope respresented
 * by "bset" given the parametric vertices and their activity domains.
 *
 * We are only interested in full-dimensional chambers.
 * Each of these chambers is the intersection of the activity domains of
 * one or more vertices and the union of all chambers is equal to the
 * projection of the entire parametric polytope onto the parameter space.
 *
 * We first create an initial chamber by intersecting as many activity
 * domains as possible without ending up with an empty or lower-dimensional
 * set.  As a minor optimization, we only consider those activity domains
 * that contain some arbitrary point.
 *
 * For each of the interior facets of the chamber, we construct a todo item,
 * containing the facet and a constraint containing the other side of the facet,
 * for constructing the chamber on the other side.
 * While their are any todo items left, we pick a todo item and
 * create the required chamber by intersecting all activity domains
 * that contain the facet and have a full-dimensional intersection with
 * the other side of the facet.  For each of the interior facets, we
 * again create todo items, taking care to cancel opposite todo items.
 */
static __isl_give isl_vertices *compute_chambers(__isl_take isl_basic_set *bset,
	__isl_take isl_vertices *vertices)
{
	int i;
	isl_ctx *ctx;
	isl_vec *sample = NULL;
	struct isl_tab *tab = NULL;
	struct isl_tab_undo *snap;
	int *selection = NULL;
	int n_chambers = 0;
	struct isl_chamber_list *list = NULL;
	struct isl_facet_todo *todo = NULL;

	if (!bset || !vertices)
		goto error;

	ctx = isl_vertices_get_ctx(vertices);
	selection = isl_alloc_array(ctx, int, vertices->n_vertices);
	if (vertices->n_vertices && !selection)
		goto error;

	bset = isl_basic_set_params(bset);

	tab = isl_tab_from_basic_set(bset, 1);
	if (!tab)
		goto error;
	for (i = 0; i < bset->n_ineq; ++i)
		if (isl_tab_freeze_constraint(tab, i) < 0)
			goto error;
	isl_basic_set_free(bset);

	snap = isl_tab_snap(tab);

	sample = isl_tab_get_sample_value(tab);

	for (i = 0; i < vertices->n_vertices; ++i) {
		selection[i] = isl_basic_set_contains(vertices->v[i].dom, sample);
		if (selection[i] < 0)
			goto error;
		if (!selection[i])
			continue;
		selection[i] = can_intersect(tab, vertices->v[i].dom);
		if (selection[i] < 0)
			goto error;
	}

	if (isl_tab_detect_redundant(tab) < 0)
		goto error;

	if (add_chamber(&list, vertices, tab, selection) < 0)
		goto error;
	n_chambers++;

	if (init_todo(&todo, tab) < 0)
		goto error;

	while (todo) {
		struct isl_facet_todo *next;

		if (isl_tab_rollback(tab, snap) < 0)
			goto error;

		if (isl_tab_add_ineq(tab, todo->constraint->el) < 0)
			goto error;
		if (isl_tab_freeze_constraint(tab, tab->n_con - 1) < 0)
			goto error;

		for (i = 0; i < vertices->n_vertices; ++i) {
			selection[i] = bset_covers_tab(vertices->v[i].dom,
							todo->tab);
			if (selection[i] < 0)
				goto error;
			if (!selection[i])
				continue;
			selection[i] = can_intersect(tab, vertices->v[i].dom);
			if (selection[i] < 0)
				goto error;
		}

		if (isl_tab_detect_redundant(tab) < 0)
			goto error;

		if (add_chamber(&list, vertices, tab, selection) < 0)
			goto error;
		n_chambers++;

		if (update_todo(todo, tab) < 0)
			goto error;

		next = todo->next;
		todo->next = NULL;
		free_todo(todo);
		todo = next;
	}

	isl_vec_free(sample);

	isl_tab_free(tab);
	free(selection);

	vertices = vertices_add_chambers(vertices, n_chambers, list);

	for (i = 0; vertices && i < vertices->n_vertices; ++i) {
		isl_basic_set_free(vertices->v[i].dom);
		vertices->v[i].dom = NULL;
	}

	return vertices;
error:
	free_chamber_list(list);
	free_todo(todo);
	isl_vec_free(sample);
	isl_tab_free(tab);
	free(selection);
	if (!tab)
		isl_basic_set_free(bset);
	isl_vertices_free(vertices);
	return NULL;
}

isl_ctx *isl_vertex_get_ctx(__isl_keep isl_vertex *vertex)
{
	return vertex ? isl_vertices_get_ctx(vertex->vertices) : NULL;
}

int isl_vertex_get_id(__isl_keep isl_vertex *vertex)
{
	return vertex ? vertex->id : -1;
}

__isl_give isl_basic_set *isl_basic_set_set_integral(__isl_take isl_basic_set *bset)
{
	if (!bset)
		return NULL;

	if (!ISL_F_ISSET(bset, ISL_BASIC_MAP_RATIONAL))
		return bset;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	ISL_F_CLR(bset, ISL_BASIC_MAP_RATIONAL);

	return isl_basic_set_finalize(bset);
}

/* Return the activity domain of the vertex "vertex".
 */
__isl_give isl_basic_set *isl_vertex_get_domain(__isl_keep isl_vertex *vertex)
{
	struct isl_vertex *v;

	if (!vertex)
		return NULL;

	v = &vertex->vertices->v[vertex->id];
	if (!v->dom) {
		v->dom = isl_basic_set_copy(v->vertex);
		v->dom = isl_basic_set_params(v->dom);
		v->dom = isl_basic_set_set_integral(v->dom);
	}

	return isl_basic_set_copy(v->dom);
}

/* Return a multiple quasi-affine expression describing the vertex "vertex"
 * in terms of the parameters,
 */
__isl_give isl_multi_aff *isl_vertex_get_expr(__isl_keep isl_vertex *vertex)
{
	struct isl_vertex *v;
	isl_basic_set *bset;

	if (!vertex)
		return NULL;

	v = &vertex->vertices->v[vertex->id];

	bset = isl_basic_set_copy(v->vertex);
	return isl_multi_aff_from_basic_set_equalities(bset);
}

static __isl_give isl_vertex *isl_vertex_alloc(__isl_take isl_vertices *vertices,
	int id)
{
	isl_ctx *ctx;
	isl_vertex *vertex;

	if (!vertices)
		return NULL;

	ctx = isl_vertices_get_ctx(vertices);
	vertex = isl_alloc_type(ctx, isl_vertex);
	if (!vertex)
		goto error;

	vertex->vertices = vertices;
	vertex->id = id;

	return vertex;
error:
	isl_vertices_free(vertices);
	return NULL;
}

void isl_vertex_free(__isl_take isl_vertex *vertex)
{
	if (!vertex)
		return;
	isl_vertices_free(vertex->vertices);
	free(vertex);
}

isl_ctx *isl_cell_get_ctx(__isl_keep isl_cell *cell)
{
	return cell ? cell->dom->ctx : NULL;
}

__isl_give isl_basic_set *isl_cell_get_domain(__isl_keep isl_cell *cell)
{
	return cell ? isl_basic_set_copy(cell->dom) : NULL;
}

static __isl_give isl_cell *isl_cell_alloc(__isl_take isl_vertices *vertices,
	__isl_take isl_basic_set *dom, int id)
{
	int i;
	isl_cell *cell = NULL;

	if (!vertices || !dom)
		goto error;

	cell = isl_calloc_type(dom->ctx, isl_cell);
	if (!cell)
		goto error;

	cell->n_vertices = vertices->c[id].n_vertices;
	cell->ids = isl_alloc_array(dom->ctx, int, cell->n_vertices);
	if (cell->n_vertices && !cell->ids)
		goto error;
	for (i = 0; i < cell->n_vertices; ++i)
		cell->ids[i] = vertices->c[id].vertices[i];
	cell->vertices = vertices;
	cell->dom = dom;

	return cell;
error:
	isl_cell_free(cell);
	isl_vertices_free(vertices);
	isl_basic_set_free(dom);
	return NULL;
}

void isl_cell_free(__isl_take isl_cell *cell)
{
	if (!cell)
		return;

	isl_vertices_free(cell->vertices);
	free(cell->ids);
	isl_basic_set_free(cell->dom);
	free(cell);
}

/* Create a tableau of the cone obtained by first homogenizing the given
 * polytope and then making all inequalities strict by setting the
 * constant term to -1.
 */
static struct isl_tab *tab_for_shifted_cone(__isl_keep isl_basic_set *bset)
{
	int i;
	isl_vec *c = NULL;
	struct isl_tab *tab;

	if (!bset)
		return NULL;
	tab = isl_tab_alloc(bset->ctx, bset->n_eq + bset->n_ineq + 1,
			    1 + isl_basic_set_total_dim(bset), 0);
	if (!tab)
		return NULL;
	tab->rational = ISL_F_ISSET(bset, ISL_BASIC_SET_RATIONAL);
	if (ISL_F_ISSET(bset, ISL_BASIC_MAP_EMPTY)) {
		if (isl_tab_mark_empty(tab) < 0)
			goto error;
		return tab;
	}

	c = isl_vec_alloc(bset->ctx, 1 + 1 + isl_basic_set_total_dim(bset));
	if (!c)
		goto error;

	isl_int_set_si(c->el[0], 0);
	for (i = 0; i < bset->n_eq; ++i) {
		isl_seq_cpy(c->el + 1, bset->eq[i], c->size - 1);
		if (isl_tab_add_eq(tab, c->el) < 0)
			goto error;
	}

	isl_int_set_si(c->el[0], -1);
	for (i = 0; i < bset->n_ineq; ++i) {
		isl_seq_cpy(c->el + 1, bset->ineq[i], c->size - 1);
		if (isl_tab_add_ineq(tab, c->el) < 0)
			goto error;
		if (tab->empty) {
			isl_vec_free(c);
			return tab;
		}
	}

	isl_seq_clr(c->el + 1, c->size - 1);
	isl_int_set_si(c->el[1], 1);
	if (isl_tab_add_ineq(tab, c->el) < 0)
		goto error;

	isl_vec_free(c);
	return tab;
error:
	isl_vec_free(c);
	isl_tab_free(tab);
	return NULL;
}

/* Compute an interior point of "bset" by selecting an interior
 * point in homogeneous space and projecting the point back down.
 */
static __isl_give isl_vec *isl_basic_set_interior_point(
	__isl_keep isl_basic_set *bset)
{
	isl_vec *vec;
	struct isl_tab *tab;

	tab = tab_for_shifted_cone(bset);
	vec = isl_tab_get_sample_value(tab);
	isl_tab_free(tab);
	if (!vec)
		return NULL;

	isl_seq_cpy(vec->el, vec->el + 1, vec->size - 1);
	vec->size--;

	return vec;
}

/* Call "fn" on all chambers of the parametric polytope with the shared
 * facets of neighboring chambers only appearing in one of the chambers.
 *
 * We pick an interior point from one of the chambers and then make
 * all constraints that do not satisfy this point strict.
 * For constraints that saturate the interior point, the sign
 * of the first non-zero coefficient is used to determine which
 * of the two (internal) constraints should be tightened.
 */
int isl_vertices_foreach_disjoint_cell(__isl_keep isl_vertices *vertices,
	int (*fn)(__isl_take isl_cell *cell, void *user), void *user)
{
	int i;
	isl_vec *vec;
	isl_cell *cell;

	if (!vertices)
		return -1;

	if (vertices->n_chambers == 0)
		return 0;

	if (vertices->n_chambers == 1) {
		isl_basic_set *dom = isl_basic_set_copy(vertices->c[0].dom);
		dom = isl_basic_set_set_integral(dom);
		cell = isl_cell_alloc(isl_vertices_copy(vertices), dom, 0);
		if (!cell)
			return -1;
		return fn(cell, user);
	}

	vec = isl_basic_set_interior_point(vertices->c[0].dom);
	if (!vec)
		return -1;

	for (i = 0; i < vertices->n_chambers; ++i) {
		int r;
		isl_basic_set *dom = isl_basic_set_copy(vertices->c[i].dom);
		if (i)
			dom = isl_basic_set_tighten_outward(dom, vec);
		dom = isl_basic_set_set_integral(dom);
		cell = isl_cell_alloc(isl_vertices_copy(vertices), dom, i);
		if (!cell)
			goto error;
		r = fn(cell, user);
		if (r < 0)
			goto error;
	}

	isl_vec_free(vec);

	return 0;
error:
	isl_vec_free(vec);
	return -1;
}

isl_stat isl_vertices_foreach_cell(__isl_keep isl_vertices *vertices,
	isl_stat (*fn)(__isl_take isl_cell *cell, void *user), void *user)
{
	int i;
	isl_cell *cell;

	if (!vertices)
		return isl_stat_error;

	if (vertices->n_chambers == 0)
		return isl_stat_ok;

	for (i = 0; i < vertices->n_chambers; ++i) {
		isl_stat r;
		isl_basic_set *dom = isl_basic_set_copy(vertices->c[i].dom);

		cell = isl_cell_alloc(isl_vertices_copy(vertices), dom, i);
		if (!cell)
			return isl_stat_error;

		r = fn(cell, user);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

isl_stat isl_vertices_foreach_vertex(__isl_keep isl_vertices *vertices,
	isl_stat (*fn)(__isl_take isl_vertex *vertex, void *user), void *user)
{
	int i;
	isl_vertex *vertex;

	if (!vertices)
		return isl_stat_error;

	if (vertices->n_vertices == 0)
		return isl_stat_ok;

	for (i = 0; i < vertices->n_vertices; ++i) {
		isl_stat r;

		vertex = isl_vertex_alloc(isl_vertices_copy(vertices), i);
		if (!vertex)
			return isl_stat_error;

		r = fn(vertex, user);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

isl_stat isl_cell_foreach_vertex(__isl_keep isl_cell *cell,
	isl_stat (*fn)(__isl_take isl_vertex *vertex, void *user), void *user)
{
	int i;
	isl_vertex *vertex;

	if (!cell)
		return isl_stat_error;

	if (cell->n_vertices == 0)
		return isl_stat_ok;

	for (i = 0; i < cell->n_vertices; ++i) {
		isl_stat r;

		vertex = isl_vertex_alloc(isl_vertices_copy(cell->vertices),
					  cell->ids[i]);
		if (!vertex)
			return isl_stat_error;

		r = fn(vertex, user);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

isl_ctx *isl_vertices_get_ctx(__isl_keep isl_vertices *vertices)
{
	return vertices ? vertices->bset->ctx : NULL;
}

int isl_vertices_get_n_vertices(__isl_keep isl_vertices *vertices)
{
	return vertices ? vertices->n_vertices : -1;
}

__isl_give isl_vertices *isl_morph_vertices(__isl_take isl_morph *morph,
	__isl_take isl_vertices *vertices)
{
	int i;
	isl_morph *param_morph = NULL;

	if (!morph || !vertices)
		goto error;

	isl_assert(vertices->bset->ctx, vertices->ref == 1, goto error);

	param_morph = isl_morph_copy(morph);
	param_morph = isl_morph_dom_params(param_morph);
	param_morph = isl_morph_ran_params(param_morph);

	for (i = 0; i < vertices->n_vertices; ++i) {
		vertices->v[i].dom = isl_morph_basic_set(
			isl_morph_copy(param_morph), vertices->v[i].dom);
		vertices->v[i].vertex = isl_morph_basic_set(
			isl_morph_copy(morph), vertices->v[i].vertex);
		if (!vertices->v[i].vertex)
			goto error;
	}

	for (i = 0; i < vertices->n_chambers; ++i) {
		vertices->c[i].dom = isl_morph_basic_set(
			isl_morph_copy(param_morph), vertices->c[i].dom);
		if (!vertices->c[i].dom)
			goto error;
	}

	isl_morph_free(param_morph);
	isl_morph_free(morph);
	return vertices;
error:
	isl_morph_free(param_morph);
	isl_morph_free(morph);
	isl_vertices_free(vertices);
	return NULL;
}

/* Construct a simplex isl_cell spanned by the vertices with indices in
 * "simplex_ids" and "other_ids" and call "fn" on this isl_cell.
 */
static int call_on_simplex(__isl_keep isl_cell *cell,
	int *simplex_ids, int n_simplex, int *other_ids, int n_other,
	int (*fn)(__isl_take isl_cell *simplex, void *user), void *user)
{
	int i;
	isl_ctx *ctx;
	struct isl_cell *simplex;

	ctx = isl_cell_get_ctx(cell);

	simplex = isl_calloc_type(ctx, struct isl_cell);
	if (!simplex)
		return -1;
	simplex->vertices = isl_vertices_copy(cell->vertices);
	if (!simplex->vertices)
		goto error;
	simplex->dom = isl_basic_set_copy(cell->dom);
	if (!simplex->dom)
		goto error;
	simplex->n_vertices = n_simplex + n_other;
	simplex->ids = isl_alloc_array(ctx, int, simplex->n_vertices);
	if (!simplex->ids)
		goto error;

	for (i = 0; i < n_simplex; ++i)
		simplex->ids[i] = simplex_ids[i];
	for (i = 0; i < n_other; ++i)
		simplex->ids[n_simplex + i] = other_ids[i];

	return fn(simplex, user);
error:
	isl_cell_free(simplex);
	return -1;
}

/* Check whether the parametric vertex described by "vertex"
 * lies on the facet corresponding to constraint "facet" of "bset".
 * The isl_vec "v" is a temporary vector than can be used by this function.
 *
 * We eliminate the variables from the facet constraint using the
 * equalities defining the vertex and check if the result is identical
 * to zero.
 *
 * It would probably be better to keep track of the constraints defining
 * a vertex during the vertex construction so that we could simply look
 * it up here.
 */
static int vertex_on_facet(__isl_keep isl_basic_set *vertex,
	__isl_keep isl_basic_set *bset, int facet, __isl_keep isl_vec *v)
{
	int i;
	isl_int m;

	isl_seq_cpy(v->el, bset->ineq[facet], v->size);

	isl_int_init(m);
	for (i = 0; i < vertex->n_eq; ++i) {
		int k = isl_seq_last_non_zero(vertex->eq[i], v->size);
		isl_seq_elim(v->el, vertex->eq[i], k, v->size, &m);
	}
	isl_int_clear(m);

	return isl_seq_first_non_zero(v->el, v->size) == -1;
}

/* Triangulate the polytope spanned by the vertices with ids
 * in "simplex_ids" and "other_ids" and call "fn" on each of
 * the resulting simplices.
 * If the input polytope is already a simplex, we simply call "fn".
 * Otherwise, we pick a point from "other_ids" and add it to "simplex_ids".
 * Then we consider each facet of "bset" that does not contain the point
 * we just picked, but does contain some of the other points in "other_ids"
 * and call ourselves recursively on the polytope spanned by the new
 * "simplex_ids" and those points in "other_ids" that lie on the facet.
 */
static int triangulate(__isl_keep isl_cell *cell, __isl_keep isl_vec *v,
	int *simplex_ids, int n_simplex, int *other_ids, int n_other,
	int (*fn)(__isl_take isl_cell *simplex, void *user), void *user)
{
	int i, j, k;
	int d, nparam;
	int *ids;
	isl_ctx *ctx;
	isl_basic_set *vertex;
	isl_basic_set *bset;

	ctx = isl_cell_get_ctx(cell);
	d = isl_basic_set_dim(cell->vertices->bset, isl_dim_set);
	nparam = isl_basic_set_dim(cell->vertices->bset, isl_dim_param);

	if (n_simplex + n_other == d + 1)
		return call_on_simplex(cell, simplex_ids, n_simplex,
				       other_ids, n_other, fn, user);

	simplex_ids[n_simplex] = other_ids[0];
	vertex = cell->vertices->v[other_ids[0]].vertex;
	bset = cell->vertices->bset;

	ids = isl_alloc_array(ctx, int, n_other - 1);
	for (i = 0; i < bset->n_ineq; ++i) {
		if (isl_seq_first_non_zero(bset->ineq[i] + 1 + nparam, d) == -1)
			continue;
		if (vertex_on_facet(vertex, bset, i, v))
			continue;

		for (j = 1, k = 0; j < n_other; ++j) {
			isl_basic_set *ov;
			ov = cell->vertices->v[other_ids[j]].vertex;
			if (vertex_on_facet(ov, bset, i, v))
				ids[k++] = other_ids[j];
		}
		if (k == 0)
			continue;

		if (triangulate(cell, v, simplex_ids, n_simplex + 1,
				ids, k, fn, user) < 0)
			goto error;
	}
	free(ids);

	return 0;
error:
	free(ids);
	return -1;
}

/* Triangulate the given cell and call "fn" on each of the resulting
 * simplices.
 */
int isl_cell_foreach_simplex(__isl_take isl_cell *cell,
	int (*fn)(__isl_take isl_cell *simplex, void *user), void *user)
{
	int d, total;
	int r;
	isl_ctx *ctx;
	isl_vec *v = NULL;
	int *simplex_ids = NULL;

	if (!cell)
		return -1;

	d = isl_basic_set_dim(cell->vertices->bset, isl_dim_set);
	total = isl_basic_set_total_dim(cell->vertices->bset);

	if (cell->n_vertices == d + 1)
		return fn(cell, user);

	ctx = isl_cell_get_ctx(cell);
	simplex_ids = isl_alloc_array(ctx, int, d + 1);
	if (!simplex_ids)
		goto error;

	v = isl_vec_alloc(ctx, 1 + total);
	if (!v)
		goto error;

	r = triangulate(cell, v, simplex_ids, 0,
			cell->ids, cell->n_vertices, fn, user);

	isl_vec_free(v);
	free(simplex_ids);

	isl_cell_free(cell);

	return r;
error:
	free(simplex_ids);
	isl_vec_free(v);
	isl_cell_free(cell);
	return -1;
}
