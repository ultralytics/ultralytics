/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 * Copyright 2015-2016 Sven Verdoolaege
 * Copyright 2016      INRIA Paris
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Centre de Recherche Inria de Paris, 2 rue Simone Iff - Voie DQ12,
 * CS 42112, 75589 Paris Cedex 12, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_space_private.h>
#include <isl_aff_private.h>
#include <isl/hash.h>
#include <isl/constraint.h>
#include <isl/schedule.h>
#include <isl_schedule_constraints.h>
#include <isl/schedule_node.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl_seq.h>
#include <isl_tab.h>
#include <isl_dim_map.h>
#include <isl/map_to_basic_set.h>
#include <isl_sort.h>
#include <isl_options_private.h>
#include <isl_tarjan.h>
#include <isl_morph.h>
#include <isl/ilp.h>
#include <isl_val_private.h>

/*
 * The scheduling algorithm implemented in this file was inspired by
 * Bondhugula et al., "Automatic Transformations for Communication-Minimized
 * Parallelization and Locality Optimization in the Polyhedral Model".
 */


/* Internal information about a node that is used during the construction
 * of a schedule.
 * space represents the space in which the domain lives
 * sched is a matrix representation of the schedule being constructed
 *	for this node; if compressed is set, then this schedule is
 *	defined over the compressed domain space
 * sched_map is an isl_map representation of the same (partial) schedule
 *	sched_map may be NULL; if compressed is set, then this map
 *	is defined over the uncompressed domain space
 * rank is the number of linearly independent rows in the linear part
 *	of sched
 * the columns of cmap represent a change of basis for the schedule
 *	coefficients; the first rank columns span the linear part of
 *	the schedule rows
 * cinv is the inverse of cmap.
 * ctrans is the transpose of cmap.
 * start is the first variable in the LP problem in the sequences that
 *	represents the schedule coefficients of this node
 * nvar is the dimension of the domain
 * nparam is the number of parameters or 0 if we are not constructing
 *	a parametric schedule
 *
 * If compressed is set, then hull represents the constraints
 * that were used to derive the compression, while compress and
 * decompress map the original space to the compressed space and
 * vice versa.
 *
 * scc is the index of SCC (or WCC) this node belongs to
 *
 * "cluster" is only used inside extract_clusters and identifies
 * the cluster of SCCs that the node belongs to.
 *
 * coincident contains a boolean for each of the rows of the schedule,
 * indicating whether the corresponding scheduling dimension satisfies
 * the coincidence constraints in the sense that the corresponding
 * dependence distances are zero.
 *
 * If the schedule_treat_coalescing option is set, then
 * "sizes" contains the sizes of the (compressed) instance set
 * in each direction.  If there is no fixed size in a given direction,
 * then the corresponding size value is set to infinity.
 * If the schedule_treat_coalescing option or the schedule_max_coefficient
 * option is set, then "max" contains the maximal values for
 * schedule coefficients of the (compressed) variables.  If no bound
 * needs to be imposed on a particular variable, then the corresponding
 * value is negative.
 */
struct isl_sched_node {
	isl_space *space;
	int	compressed;
	isl_set	*hull;
	isl_multi_aff *compress;
	isl_multi_aff *decompress;
	isl_mat *sched;
	isl_map *sched_map;
	int	 rank;
	isl_mat *cmap;
	isl_mat *cinv;
	isl_mat *ctrans;
	int	 start;
	int	 nvar;
	int	 nparam;

	int	 scc;
	int	 cluster;

	int	*coincident;

	isl_multi_val *sizes;
	isl_vec *max;
};

static int node_has_space(const void *entry, const void *val)
{
	struct isl_sched_node *node = (struct isl_sched_node *)entry;
	isl_space *dim = (isl_space *)val;

	return isl_space_is_equal(node->space, dim);
}

static int node_scc_exactly(struct isl_sched_node *node, int scc)
{
	return node->scc == scc;
}

static int node_scc_at_most(struct isl_sched_node *node, int scc)
{
	return node->scc <= scc;
}

static int node_scc_at_least(struct isl_sched_node *node, int scc)
{
	return node->scc >= scc;
}

/* An edge in the dependence graph.  An edge may be used to
 * ensure validity of the generated schedule, to minimize the dependence
 * distance or both
 *
 * map is the dependence relation, with i -> j in the map if j depends on i
 * tagged_condition and tagged_validity contain the union of all tagged
 *	condition or conditional validity dependence relations that
 *	specialize the dependence relation "map"; that is,
 *	if (i -> a) -> (j -> b) is an element of "tagged_condition"
 *	or "tagged_validity", then i -> j is an element of "map".
 *	If these fields are NULL, then they represent the empty relation.
 * src is the source node
 * dst is the sink node
 *
 * types is a bit vector containing the types of this edge.
 * validity is set if the edge is used to ensure correctness
 * coincidence is used to enforce zero dependence distances
 * proximity is set if the edge is used to minimize dependence distances
 * condition is set if the edge represents a condition
 *	for a conditional validity schedule constraint
 * local can only be set for condition edges and indicates that
 *	the dependence distance over the edge should be zero
 * conditional_validity is set if the edge is used to conditionally
 *	ensure correctness
 *
 * For validity edges, start and end mark the sequence of inequality
 * constraints in the LP problem that encode the validity constraint
 * corresponding to this edge.
 *
 * During clustering, an edge may be marked "no_merge" if it should
 * not be used to merge clusters.
 * The weight is also only used during clustering and it is
 * an indication of how many schedule dimensions on either side
 * of the schedule constraints can be aligned.
 * If the weight is negative, then this means that this edge was postponed
 * by has_bounded_distances or any_no_merge.  The original weight can
 * be retrieved by adding 1 + graph->max_weight, with "graph"
 * the graph containing this edge.
 */
struct isl_sched_edge {
	isl_map *map;
	isl_union_map *tagged_condition;
	isl_union_map *tagged_validity;

	struct isl_sched_node *src;
	struct isl_sched_node *dst;

	unsigned types;

	int start;
	int end;

	int no_merge;
	int weight;
};

/* Is "edge" marked as being of type "type"?
 */
static int is_type(struct isl_sched_edge *edge, enum isl_edge_type type)
{
	return ISL_FL_ISSET(edge->types, 1 << type);
}

/* Mark "edge" as being of type "type".
 */
static void set_type(struct isl_sched_edge *edge, enum isl_edge_type type)
{
	ISL_FL_SET(edge->types, 1 << type);
}

/* No longer mark "edge" as being of type "type"?
 */
static void clear_type(struct isl_sched_edge *edge, enum isl_edge_type type)
{
	ISL_FL_CLR(edge->types, 1 << type);
}

/* Is "edge" marked as a validity edge?
 */
static int is_validity(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_validity);
}

/* Mark "edge" as a validity edge.
 */
static void set_validity(struct isl_sched_edge *edge)
{
	set_type(edge, isl_edge_validity);
}

/* Is "edge" marked as a proximity edge?
 */
static int is_proximity(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_proximity);
}

/* Is "edge" marked as a local edge?
 */
static int is_local(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_local);
}

/* Mark "edge" as a local edge.
 */
static void set_local(struct isl_sched_edge *edge)
{
	set_type(edge, isl_edge_local);
}

/* No longer mark "edge" as a local edge.
 */
static void clear_local(struct isl_sched_edge *edge)
{
	clear_type(edge, isl_edge_local);
}

/* Is "edge" marked as a coincidence edge?
 */
static int is_coincidence(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_coincidence);
}

/* Is "edge" marked as a condition edge?
 */
static int is_condition(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_condition);
}

/* Is "edge" marked as a conditional validity edge?
 */
static int is_conditional_validity(struct isl_sched_edge *edge)
{
	return is_type(edge, isl_edge_conditional_validity);
}

/* Internal information about the dependence graph used during
 * the construction of the schedule.
 *
 * intra_hmap is a cache, mapping dependence relations to their dual,
 *	for dependences from a node to itself
 * inter_hmap is a cache, mapping dependence relations to their dual,
 *	for dependences between distinct nodes
 * if compression is involved then the key for these maps
 * is the original, uncompressed dependence relation, while
 * the value is the dual of the compressed dependence relation.
 *
 * n is the number of nodes
 * node is the list of nodes
 * maxvar is the maximal number of variables over all nodes
 * max_row is the allocated number of rows in the schedule
 * n_row is the current (maximal) number of linearly independent
 *	rows in the node schedules
 * n_total_row is the current number of rows in the node schedules
 * band_start is the starting row in the node schedules of the current band
 * root is set if this graph is the original dependence graph,
 *	without any splitting
 *
 * sorted contains a list of node indices sorted according to the
 *	SCC to which a node belongs
 *
 * n_edge is the number of edges
 * edge is the list of edges
 * max_edge contains the maximal number of edges of each type;
 *	in particular, it contains the number of edges in the inital graph.
 * edge_table contains pointers into the edge array, hashed on the source
 *	and sink spaces; there is one such table for each type;
 *	a given edge may be referenced from more than one table
 *	if the corresponding relation appears in more than one of the
 *	sets of dependences; however, for each type there is only
 *	a single edge between a given pair of source and sink space
 *	in the entire graph
 *
 * node_table contains pointers into the node array, hashed on the space
 *
 * region contains a list of variable sequences that should be non-trivial
 *
 * lp contains the (I)LP problem used to obtain new schedule rows
 *
 * src_scc and dst_scc are the source and sink SCCs of an edge with
 *	conflicting constraints
 *
 * scc represents the number of components
 * weak is set if the components are weakly connected
 *
 * max_weight is used during clustering and represents the maximal
 * weight of the relevant proximity edges.
 */
struct isl_sched_graph {
	isl_map_to_basic_set *intra_hmap;
	isl_map_to_basic_set *inter_hmap;

	struct isl_sched_node *node;
	int n;
	int maxvar;
	int max_row;
	int n_row;

	int *sorted;

	int n_total_row;
	int band_start;

	int root;

	struct isl_sched_edge *edge;
	int n_edge;
	int max_edge[isl_edge_last + 1];
	struct isl_hash_table *edge_table[isl_edge_last + 1];

	struct isl_hash_table *node_table;
	struct isl_region *region;

	isl_basic_set *lp;

	int src_scc;
	int dst_scc;

	int scc;
	int weak;

	int max_weight;
};

/* Initialize node_table based on the list of nodes.
 */
static int graph_init_table(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	graph->node_table = isl_hash_table_alloc(ctx, graph->n);
	if (!graph->node_table)
		return -1;

	for (i = 0; i < graph->n; ++i) {
		struct isl_hash_table_entry *entry;
		uint32_t hash;

		hash = isl_space_get_hash(graph->node[i].space);
		entry = isl_hash_table_find(ctx, graph->node_table, hash,
					    &node_has_space,
					    graph->node[i].space, 1);
		if (!entry)
			return -1;
		entry->data = &graph->node[i];
	}

	return 0;
}

/* Return a pointer to the node that lives within the given space,
 * or NULL if there is no such node.
 */
static struct isl_sched_node *graph_find_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_space *dim)
{
	struct isl_hash_table_entry *entry;
	uint32_t hash;

	hash = isl_space_get_hash(dim);
	entry = isl_hash_table_find(ctx, graph->node_table, hash,
				    &node_has_space, dim, 0);

	return entry ? entry->data : NULL;
}

static int edge_has_src_and_dst(const void *entry, const void *val)
{
	const struct isl_sched_edge *edge = entry;
	const struct isl_sched_edge *temp = val;

	return edge->src == temp->src && edge->dst == temp->dst;
}

/* Add the given edge to graph->edge_table[type].
 */
static isl_stat graph_edge_table_add(isl_ctx *ctx,
	struct isl_sched_graph *graph, enum isl_edge_type type,
	struct isl_sched_edge *edge)
{
	struct isl_hash_table_entry *entry;
	uint32_t hash;

	hash = isl_hash_init();
	hash = isl_hash_builtin(hash, edge->src);
	hash = isl_hash_builtin(hash, edge->dst);
	entry = isl_hash_table_find(ctx, graph->edge_table[type], hash,
				    &edge_has_src_and_dst, edge, 1);
	if (!entry)
		return isl_stat_error;
	entry->data = edge;

	return isl_stat_ok;
}

/* Allocate the edge_tables based on the maximal number of edges of
 * each type.
 */
static int graph_init_edge_tables(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i <= isl_edge_last; ++i) {
		graph->edge_table[i] = isl_hash_table_alloc(ctx,
							    graph->max_edge[i]);
		if (!graph->edge_table[i])
			return -1;
	}

	return 0;
}

/* If graph->edge_table[type] contains an edge from the given source
 * to the given destination, then return the hash table entry of this edge.
 * Otherwise, return NULL.
 */
static struct isl_hash_table_entry *graph_find_edge_entry(
	struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	isl_ctx *ctx = isl_space_get_ctx(src->space);
	uint32_t hash;
	struct isl_sched_edge temp = { .src = src, .dst = dst };

	hash = isl_hash_init();
	hash = isl_hash_builtin(hash, temp.src);
	hash = isl_hash_builtin(hash, temp.dst);
	return isl_hash_table_find(ctx, graph->edge_table[type], hash,
				    &edge_has_src_and_dst, &temp, 0);
}


/* If graph->edge_table[type] contains an edge from the given source
 * to the given destination, then return this edge.
 * Otherwise, return NULL.
 */
static struct isl_sched_edge *graph_find_edge(struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	struct isl_hash_table_entry *entry;

	entry = graph_find_edge_entry(graph, type, src, dst);
	if (!entry)
		return NULL;

	return entry->data;
}

/* Check whether the dependence graph has an edge of the given type
 * between the given two nodes.
 */
static isl_bool graph_has_edge(struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	struct isl_sched_edge *edge;
	isl_bool empty;

	edge = graph_find_edge(graph, type, src, dst);
	if (!edge)
		return 0;

	empty = isl_map_plain_is_empty(edge->map);
	if (empty < 0)
		return isl_bool_error;

	return !empty;
}

/* Look for any edge with the same src, dst and map fields as "model".
 *
 * Return the matching edge if one can be found.
 * Return "model" if no matching edge is found.
 * Return NULL on error.
 */
static struct isl_sched_edge *graph_find_matching_edge(
	struct isl_sched_graph *graph, struct isl_sched_edge *model)
{
	enum isl_edge_type i;
	struct isl_sched_edge *edge;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		int is_equal;

		edge = graph_find_edge(graph, i, model->src, model->dst);
		if (!edge)
			continue;
		is_equal = isl_map_plain_is_equal(model->map, edge->map);
		if (is_equal < 0)
			return NULL;
		if (is_equal)
			return edge;
	}

	return model;
}

/* Remove the given edge from all the edge_tables that refer to it.
 */
static void graph_remove_edge(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	isl_ctx *ctx = isl_map_get_ctx(edge->map);
	enum isl_edge_type i;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		struct isl_hash_table_entry *entry;

		entry = graph_find_edge_entry(graph, i, edge->src, edge->dst);
		if (!entry)
			continue;
		if (entry->data != edge)
			continue;
		isl_hash_table_remove(ctx, graph->edge_table[i], entry);
	}
}

/* Check whether the dependence graph has any edge
 * between the given two nodes.
 */
static isl_bool graph_has_any_edge(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	enum isl_edge_type i;
	isl_bool r;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		r = graph_has_edge(graph, i, src, dst);
		if (r < 0 || r)
			return r;
	}

	return r;
}

/* Check whether the dependence graph has a validity edge
 * between the given two nodes.
 *
 * Conditional validity edges are essentially validity edges that
 * can be ignored if the corresponding condition edges are iteration private.
 * Here, we are only checking for the presence of validity
 * edges, so we need to consider the conditional validity edges too.
 * In particular, this function is used during the detection
 * of strongly connected components and we cannot ignore
 * conditional validity edges during this detection.
 */
static isl_bool graph_has_validity_edge(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	isl_bool r;

	r = graph_has_edge(graph, isl_edge_validity, src, dst);
	if (r < 0 || r)
		return r;

	return graph_has_edge(graph, isl_edge_conditional_validity, src, dst);
}

static int graph_alloc(isl_ctx *ctx, struct isl_sched_graph *graph,
	int n_node, int n_edge)
{
	int i;

	graph->n = n_node;
	graph->n_edge = n_edge;
	graph->node = isl_calloc_array(ctx, struct isl_sched_node, graph->n);
	graph->sorted = isl_calloc_array(ctx, int, graph->n);
	graph->region = isl_alloc_array(ctx, struct isl_region, graph->n);
	graph->edge = isl_calloc_array(ctx,
					struct isl_sched_edge, graph->n_edge);

	graph->intra_hmap = isl_map_to_basic_set_alloc(ctx, 2 * n_edge);
	graph->inter_hmap = isl_map_to_basic_set_alloc(ctx, 2 * n_edge);

	if (!graph->node || !graph->region || (graph->n_edge && !graph->edge) ||
	    !graph->sorted)
		return -1;

	for(i = 0; i < graph->n; ++i)
		graph->sorted[i] = i;

	return 0;
}

static void graph_free(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	isl_map_to_basic_set_free(graph->intra_hmap);
	isl_map_to_basic_set_free(graph->inter_hmap);

	if (graph->node)
		for (i = 0; i < graph->n; ++i) {
			isl_space_free(graph->node[i].space);
			isl_set_free(graph->node[i].hull);
			isl_multi_aff_free(graph->node[i].compress);
			isl_multi_aff_free(graph->node[i].decompress);
			isl_mat_free(graph->node[i].sched);
			isl_map_free(graph->node[i].sched_map);
			isl_mat_free(graph->node[i].cmap);
			isl_mat_free(graph->node[i].cinv);
			isl_mat_free(graph->node[i].ctrans);
			if (graph->root)
				free(graph->node[i].coincident);
			isl_multi_val_free(graph->node[i].sizes);
			isl_vec_free(graph->node[i].max);
		}
	free(graph->node);
	free(graph->sorted);
	if (graph->edge)
		for (i = 0; i < graph->n_edge; ++i) {
			isl_map_free(graph->edge[i].map);
			isl_union_map_free(graph->edge[i].tagged_condition);
			isl_union_map_free(graph->edge[i].tagged_validity);
		}
	free(graph->edge);
	free(graph->region);
	for (i = 0; i <= isl_edge_last; ++i)
		isl_hash_table_free(ctx, graph->edge_table[i]);
	isl_hash_table_free(ctx, graph->node_table);
	isl_basic_set_free(graph->lp);
}

/* For each "set" on which this function is called, increment
 * graph->n by one and update graph->maxvar.
 */
static isl_stat init_n_maxvar(__isl_take isl_set *set, void *user)
{
	struct isl_sched_graph *graph = user;
	int nvar = isl_set_dim(set, isl_dim_set);

	graph->n++;
	if (nvar > graph->maxvar)
		graph->maxvar = nvar;

	isl_set_free(set);

	return isl_stat_ok;
}

/* Compute the number of rows that should be allocated for the schedule.
 * In particular, we need one row for each variable or one row
 * for each basic map in the dependences.
 * Note that it is practically impossible to exhaust both
 * the number of dependences and the number of variables.
 */
static isl_stat compute_max_row(struct isl_sched_graph *graph,
	__isl_keep isl_schedule_constraints *sc)
{
	int n_edge;
	isl_stat r;
	isl_union_set *domain;

	graph->n = 0;
	graph->maxvar = 0;
	domain = isl_schedule_constraints_get_domain(sc);
	r = isl_union_set_foreach_set(domain, &init_n_maxvar, graph);
	isl_union_set_free(domain);
	if (r < 0)
		return isl_stat_error;
	n_edge = isl_schedule_constraints_n_basic_map(sc);
	if (n_edge < 0)
		return isl_stat_error;
	graph->max_row = n_edge + graph->maxvar;

	return isl_stat_ok;
}

/* Does "bset" have any defining equalities for its set variables?
 */
static int has_any_defining_equality(__isl_keep isl_basic_set *bset)
{
	int i, n;

	if (!bset)
		return -1;

	n = isl_basic_set_dim(bset, isl_dim_set);
	for (i = 0; i < n; ++i) {
		int has;

		has = isl_basic_set_has_defining_equality(bset, isl_dim_set, i,
							NULL);
		if (has < 0 || has)
			return has;
	}

	return 0;
}

/* Set the entries of node->max to the value of the schedule_max_coefficient
 * option, if set.
 */
static isl_stat set_max_coefficient(isl_ctx *ctx, struct isl_sched_node *node)
{
	int max;

	max = isl_options_get_schedule_max_coefficient(ctx);
	if (max == -1)
		return isl_stat_ok;

	node->max = isl_vec_alloc(ctx, node->nvar);
	node->max = isl_vec_set_si(node->max, max);
	if (!node->max)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Set the entries of node->max to the minimum of the schedule_max_coefficient
 * option (if set) and half of the minimum of the sizes in the other
 * dimensions.  If the minimum of the sizes is one, half of the size
 * is zero and this value is reset to one.
 * If the global minimum is unbounded (i.e., if both
 * the schedule_max_coefficient is not set and the sizes in the other
 * dimensions are unbounded), then store a negative value.
 * If the schedule coefficient is close to the size of the instance set
 * in another dimension, then the schedule may represent a loop
 * coalescing transformation (especially if the coefficient
 * in that other dimension is one).  Forcing the coefficient to be
 * smaller than or equal to half the minimal size should avoid this
 * situation.
 */
static isl_stat compute_max_coefficient(isl_ctx *ctx,
	struct isl_sched_node *node)
{
	int max;
	int i, j;
	isl_vec *v;

	max = isl_options_get_schedule_max_coefficient(ctx);
	v = isl_vec_alloc(ctx, node->nvar);
	if (!v)
		return isl_stat_error;

	for (i = 0; i < node->nvar; ++i) {
		isl_int_set_si(v->el[i], max);
		isl_int_mul_si(v->el[i], v->el[i], 2);
	}

	for (i = 0; i < node->nvar; ++i) {
		isl_val *size;

		size = isl_multi_val_get_val(node->sizes, i);
		if (!size)
			goto error;
		if (!isl_val_is_int(size)) {
			isl_val_free(size);
			continue;
		}
		for (j = 0; j < node->nvar; ++j) {
			if (j == i)
				continue;
			if (isl_int_is_neg(v->el[j]) ||
			    isl_int_gt(v->el[j], size->n))
				isl_int_set(v->el[j], size->n);
		}
		isl_val_free(size);
	}

	for (i = 0; i < node->nvar; ++i) {
		isl_int_fdiv_q_ui(v->el[i], v->el[i], 2);
		if (isl_int_is_zero(v->el[i]))
			isl_int_set_si(v->el[i], 1);
	}

	node->max = v;
	return isl_stat_ok;
error:
	isl_vec_free(v);
	return isl_stat_error;
}

/* Compute and return the size of "set" in dimension "dim".
 * The size is taken to be the difference in values for that variable
 * for fixed values of the other variables.
 * In particular, the variable is first isolated from the other variables
 * in the range of a map
 *
 *	[i_0, ..., i_dim-1, i_dim+1, ...] -> [i_dim]
 *
 * and then duplicated
 *
 *	[i_0, ..., i_dim-1, i_dim+1, ...] -> [[i_dim] -> [i_dim']]
 *
 * The shared variables are then projected out and the maximal value
 * of i_dim' - i_dim is computed.
 */
static __isl_give isl_val *compute_size(__isl_take isl_set *set, int dim)
{
	isl_map *map;
	isl_local_space *ls;
	isl_aff *obj;
	isl_val *v;

	map = isl_set_project_onto_map(set, isl_dim_set, dim, 1);
	map = isl_map_project_out(map, isl_dim_in, dim, 1);
	map = isl_map_range_product(map, isl_map_copy(map));
	map = isl_set_unwrap(isl_map_range(map));
	set = isl_map_deltas(map);
	ls = isl_local_space_from_space(isl_set_get_space(set));
	obj = isl_aff_var_on_domain(ls, isl_dim_set, 0);
	v = isl_set_max_val(set, obj);
	isl_aff_free(obj);
	isl_set_free(set);

	return v;
}

/* Compute the size of the instance set "set" of "node", after compression,
 * as well as bounds on the corresponding coefficients, if needed.
 *
 * The sizes are needed when the schedule_treat_coalescing option is set.
 * The bounds are needed when the schedule_treat_coalescing option or
 * the schedule_max_coefficient option is set.
 *
 * If the schedule_treat_coalescing option is not set, then at most
 * the bounds need to be set and this is done in set_max_coefficient.
 * Otherwise, compress the domain if needed, compute the size
 * in each direction and store the results in node->size.
 * Finally, set the bounds on the coefficients based on the sizes
 * and the schedule_max_coefficient option in compute_max_coefficient.
 */
static isl_stat compute_sizes_and_max(isl_ctx *ctx, struct isl_sched_node *node,
	__isl_take isl_set *set)
{
	int j, n;
	isl_multi_val *mv;

	if (!isl_options_get_schedule_treat_coalescing(ctx)) {
		isl_set_free(set);
		return set_max_coefficient(ctx, node);
	}

	if (node->compressed)
		set = isl_set_preimage_multi_aff(set,
					isl_multi_aff_copy(node->decompress));
	mv = isl_multi_val_zero(isl_set_get_space(set));
	n = isl_set_dim(set, isl_dim_set);
	for (j = 0; j < n; ++j) {
		isl_val *v;

		v = compute_size(isl_set_copy(set), j);
		mv = isl_multi_val_set_val(mv, j, v);
	}
	node->sizes = mv;
	isl_set_free(set);
	if (!node->sizes)
		return isl_stat_error;
	return compute_max_coefficient(ctx, node);
}

/* Add a new node to the graph representing the given instance set.
 * "nvar" is the (possibly compressed) number of variables and
 * may be smaller than then number of set variables in "set"
 * if "compressed" is set.
 * If "compressed" is set, then "hull" represents the constraints
 * that were used to derive the compression, while "compress" and
 * "decompress" map the original space to the compressed space and
 * vice versa.
 * If "compressed" is not set, then "hull", "compress" and "decompress"
 * should be NULL.
 *
 * Compute the size of the instance set and bounds on the coefficients,
 * if needed.
 */
static isl_stat add_node(struct isl_sched_graph *graph,
	__isl_take isl_set *set, int nvar, int compressed,
	__isl_take isl_set *hull, __isl_take isl_multi_aff *compress,
	__isl_take isl_multi_aff *decompress)
{
	int nparam;
	isl_ctx *ctx;
	isl_mat *sched;
	isl_space *space;
	int *coincident;
	struct isl_sched_node *node;

	if (!set)
		return isl_stat_error;

	ctx = isl_set_get_ctx(set);
	nparam = isl_set_dim(set, isl_dim_param);
	if (!ctx->opt->schedule_parametric)
		nparam = 0;
	sched = isl_mat_alloc(ctx, 0, 1 + nparam + nvar);
	node = &graph->node[graph->n];
	graph->n++;
	space = isl_set_get_space(set);
	node->space = space;
	node->nvar = nvar;
	node->nparam = nparam;
	node->sched = sched;
	node->sched_map = NULL;
	coincident = isl_calloc_array(ctx, int, graph->max_row);
	node->coincident = coincident;
	node->compressed = compressed;
	node->hull = hull;
	node->compress = compress;
	node->decompress = decompress;
	if (compute_sizes_and_max(ctx, node, set) < 0)
		return isl_stat_error;

	if (!space || !sched || (graph->max_row && !coincident))
		return isl_stat_error;
	if (compressed && (!hull || !compress || !decompress))
		return isl_stat_error;

	return isl_stat_ok;
}

/* Add a new node to the graph representing the given set.
 *
 * If any of the set variables is defined by an equality, then
 * we perform variable compression such that we can perform
 * the scheduling on the compressed domain.
 */
static isl_stat extract_node(__isl_take isl_set *set, void *user)
{
	int nvar;
	int has_equality;
	isl_basic_set *hull;
	isl_set *hull_set;
	isl_morph *morph;
	isl_multi_aff *compress, *decompress;
	struct isl_sched_graph *graph = user;

	hull = isl_set_affine_hull(isl_set_copy(set));
	hull = isl_basic_set_remove_divs(hull);
	nvar = isl_set_dim(set, isl_dim_set);
	has_equality = has_any_defining_equality(hull);

	if (has_equality < 0)
		goto error;
	if (!has_equality) {
		isl_basic_set_free(hull);
		return add_node(graph, set, nvar, 0, NULL, NULL, NULL);
	}

	morph = isl_basic_set_variable_compression(hull, isl_dim_set);
	nvar = isl_morph_ran_dim(morph, isl_dim_set);
	compress = isl_morph_get_var_multi_aff(morph);
	morph = isl_morph_inverse(morph);
	decompress = isl_morph_get_var_multi_aff(morph);
	isl_morph_free(morph);

	hull_set = isl_set_from_basic_set(hull);
	return add_node(graph, set, nvar, 1, hull_set, compress, decompress);
error:
	isl_basic_set_free(hull);
	isl_set_free(set);
	return isl_stat_error;
}

struct isl_extract_edge_data {
	enum isl_edge_type type;
	struct isl_sched_graph *graph;
};

/* Merge edge2 into edge1, freeing the contents of edge2.
 * Return 0 on success and -1 on failure.
 *
 * edge1 and edge2 are assumed to have the same value for the map field.
 */
static int merge_edge(struct isl_sched_edge *edge1,
	struct isl_sched_edge *edge2)
{
	edge1->types |= edge2->types;
	isl_map_free(edge2->map);

	if (is_condition(edge2)) {
		if (!edge1->tagged_condition)
			edge1->tagged_condition = edge2->tagged_condition;
		else
			edge1->tagged_condition =
				isl_union_map_union(edge1->tagged_condition,
						    edge2->tagged_condition);
	}

	if (is_conditional_validity(edge2)) {
		if (!edge1->tagged_validity)
			edge1->tagged_validity = edge2->tagged_validity;
		else
			edge1->tagged_validity =
				isl_union_map_union(edge1->tagged_validity,
						    edge2->tagged_validity);
	}

	if (is_condition(edge2) && !edge1->tagged_condition)
		return -1;
	if (is_conditional_validity(edge2) && !edge1->tagged_validity)
		return -1;

	return 0;
}

/* Insert dummy tags in domain and range of "map".
 *
 * In particular, if "map" is of the form
 *
 *	A -> B
 *
 * then return
 *
 *	[A -> dummy_tag] -> [B -> dummy_tag]
 *
 * where the dummy_tags are identical and equal to any dummy tags
 * introduced by any other call to this function.
 */
static __isl_give isl_map *insert_dummy_tags(__isl_take isl_map *map)
{
	static char dummy;
	isl_ctx *ctx;
	isl_id *id;
	isl_space *space;
	isl_set *domain, *range;

	ctx = isl_map_get_ctx(map);

	id = isl_id_alloc(ctx, NULL, &dummy);
	space = isl_space_params(isl_map_get_space(map));
	space = isl_space_set_from_params(space);
	space = isl_space_set_tuple_id(space, isl_dim_set, id);
	space = isl_space_map_from_set(space);

	domain = isl_map_wrap(map);
	range = isl_map_wrap(isl_map_universe(space));
	map = isl_map_from_domain_and_range(domain, range);
	map = isl_map_zip(map);

	return map;
}

/* Given that at least one of "src" or "dst" is compressed, return
 * a map between the spaces of these nodes restricted to the affine
 * hull that was used in the compression.
 */
static __isl_give isl_map *extract_hull(struct isl_sched_node *src,
	struct isl_sched_node *dst)
{
	isl_set *dom, *ran;

	if (src->compressed)
		dom = isl_set_copy(src->hull);
	else
		dom = isl_set_universe(isl_space_copy(src->space));
	if (dst->compressed)
		ran = isl_set_copy(dst->hull);
	else
		ran = isl_set_universe(isl_space_copy(dst->space));

	return isl_map_from_domain_and_range(dom, ran);
}

/* Intersect the domains of the nested relations in domain and range
 * of "tagged" with "map".
 */
static __isl_give isl_map *map_intersect_domains(__isl_take isl_map *tagged,
	__isl_keep isl_map *map)
{
	isl_set *set;

	tagged = isl_map_zip(tagged);
	set = isl_map_wrap(isl_map_copy(map));
	tagged = isl_map_intersect_domain(tagged, set);
	tagged = isl_map_zip(tagged);
	return tagged;
}

/* Return a pointer to the node that lives in the domain space of "map"
 * or NULL if there is no such node.
 */
static struct isl_sched_node *find_domain_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_map *map)
{
	struct isl_sched_node *node;
	isl_space *space;

	space = isl_space_domain(isl_map_get_space(map));
	node = graph_find_node(ctx, graph, space);
	isl_space_free(space);

	return node;
}

/* Return a pointer to the node that lives in the range space of "map"
 * or NULL if there is no such node.
 */
static struct isl_sched_node *find_range_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_map *map)
{
	struct isl_sched_node *node;
	isl_space *space;

	space = isl_space_range(isl_map_get_space(map));
	node = graph_find_node(ctx, graph, space);
	isl_space_free(space);

	return node;
}

/* Add a new edge to the graph based on the given map
 * and add it to data->graph->edge_table[data->type].
 * If a dependence relation of a given type happens to be identical
 * to one of the dependence relations of a type that was added before,
 * then we don't create a new edge, but instead mark the original edge
 * as also representing a dependence of the current type.
 *
 * Edges of type isl_edge_condition or isl_edge_conditional_validity
 * may be specified as "tagged" dependence relations.  That is, "map"
 * may contain elements (i -> a) -> (j -> b), where i -> j denotes
 * the dependence on iterations and a and b are tags.
 * edge->map is set to the relation containing the elements i -> j,
 * while edge->tagged_condition and edge->tagged_validity contain
 * the union of all the "map" relations
 * for which extract_edge is called that result in the same edge->map.
 *
 * If the source or the destination node is compressed, then
 * intersect both "map" and "tagged" with the constraints that
 * were used to construct the compression.
 * This ensures that there are no schedule constraints defined
 * outside of these domains, while the scheduler no longer has
 * any control over those outside parts.
 */
static isl_stat extract_edge(__isl_take isl_map *map, void *user)
{
	isl_ctx *ctx = isl_map_get_ctx(map);
	struct isl_extract_edge_data *data = user;
	struct isl_sched_graph *graph = data->graph;
	struct isl_sched_node *src, *dst;
	struct isl_sched_edge *edge;
	isl_map *tagged = NULL;

	if (data->type == isl_edge_condition ||
	    data->type == isl_edge_conditional_validity) {
		if (isl_map_can_zip(map)) {
			tagged = isl_map_copy(map);
			map = isl_set_unwrap(isl_map_domain(isl_map_zip(map)));
		} else {
			tagged = insert_dummy_tags(isl_map_copy(map));
		}
	}

	src = find_domain_node(ctx, graph, map);
	dst = find_range_node(ctx, graph, map);

	if (!src || !dst) {
		isl_map_free(map);
		isl_map_free(tagged);
		return isl_stat_ok;
	}

	if (src->compressed || dst->compressed) {
		isl_map *hull;
		hull = extract_hull(src, dst);
		if (tagged)
			tagged = map_intersect_domains(tagged, hull);
		map = isl_map_intersect(map, hull);
	}

	graph->edge[graph->n_edge].src = src;
	graph->edge[graph->n_edge].dst = dst;
	graph->edge[graph->n_edge].map = map;
	graph->edge[graph->n_edge].types = 0;
	graph->edge[graph->n_edge].tagged_condition = NULL;
	graph->edge[graph->n_edge].tagged_validity = NULL;
	set_type(&graph->edge[graph->n_edge], data->type);
	if (data->type == isl_edge_condition)
		graph->edge[graph->n_edge].tagged_condition =
					isl_union_map_from_map(tagged);
	if (data->type == isl_edge_conditional_validity)
		graph->edge[graph->n_edge].tagged_validity =
					isl_union_map_from_map(tagged);

	edge = graph_find_matching_edge(graph, &graph->edge[graph->n_edge]);
	if (!edge) {
		graph->n_edge++;
		return isl_stat_error;
	}
	if (edge == &graph->edge[graph->n_edge])
		return graph_edge_table_add(ctx, graph, data->type,
				    &graph->edge[graph->n_edge++]);

	if (merge_edge(edge, &graph->edge[graph->n_edge]) < 0)
		return -1;

	return graph_edge_table_add(ctx, graph, data->type, edge);
}

/* Initialize the schedule graph "graph" from the schedule constraints "sc".
 *
 * The context is included in the domain before the nodes of
 * the graphs are extracted in order to be able to exploit
 * any possible additional equalities.
 * Note that this intersection is only performed locally here.
 */
static isl_stat graph_init(struct isl_sched_graph *graph,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_ctx *ctx;
	isl_union_set *domain;
	isl_union_map *c;
	struct isl_extract_edge_data data;
	enum isl_edge_type i;
	isl_stat r;

	if (!sc)
		return isl_stat_error;

	ctx = isl_schedule_constraints_get_ctx(sc);

	domain = isl_schedule_constraints_get_domain(sc);
	graph->n = isl_union_set_n_set(domain);
	isl_union_set_free(domain);

	if (graph_alloc(ctx, graph, graph->n,
	    isl_schedule_constraints_n_map(sc)) < 0)
		return isl_stat_error;

	if (compute_max_row(graph, sc) < 0)
		return isl_stat_error;
	graph->root = 1;
	graph->n = 0;
	domain = isl_schedule_constraints_get_domain(sc);
	domain = isl_union_set_intersect_params(domain,
				    isl_schedule_constraints_get_context(sc));
	r = isl_union_set_foreach_set(domain, &extract_node, graph);
	isl_union_set_free(domain);
	if (r < 0)
		return isl_stat_error;
	if (graph_init_table(ctx, graph) < 0)
		return isl_stat_error;
	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		c = isl_schedule_constraints_get(sc, i);
		graph->max_edge[i] = isl_union_map_n_map(c);
		isl_union_map_free(c);
		if (!c)
			return isl_stat_error;
	}
	if (graph_init_edge_tables(ctx, graph) < 0)
		return isl_stat_error;
	graph->n_edge = 0;
	data.graph = graph;
	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		isl_stat r;

		data.type = i;
		c = isl_schedule_constraints_get(sc, i);
		r = isl_union_map_foreach_map(c, &extract_edge, &data);
		isl_union_map_free(c);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Check whether there is any dependence from node[j] to node[i]
 * or from node[i] to node[j].
 */
static isl_bool node_follows_weak(int i, int j, void *user)
{
	isl_bool f;
	struct isl_sched_graph *graph = user;

	f = graph_has_any_edge(graph, &graph->node[j], &graph->node[i]);
	if (f < 0 || f)
		return f;
	return graph_has_any_edge(graph, &graph->node[i], &graph->node[j]);
}

/* Check whether there is a (conditional) validity dependence from node[j]
 * to node[i], forcing node[i] to follow node[j].
 */
static isl_bool node_follows_strong(int i, int j, void *user)
{
	struct isl_sched_graph *graph = user;

	return graph_has_validity_edge(graph, &graph->node[j], &graph->node[i]);
}

/* Use Tarjan's algorithm for computing the strongly connected components
 * in the dependence graph only considering those edges defined by "follows".
 */
static int detect_ccs(isl_ctx *ctx, struct isl_sched_graph *graph,
	isl_bool (*follows)(int i, int j, void *user))
{
	int i, n;
	struct isl_tarjan_graph *g = NULL;

	g = isl_tarjan_graph_init(ctx, graph->n, follows, graph);
	if (!g)
		return -1;

	graph->scc = 0;
	i = 0;
	n = graph->n;
	while (n) {
		while (g->order[i] != -1) {
			graph->node[g->order[i]].scc = graph->scc;
			--n;
			++i;
		}
		++i;
		graph->scc++;
	}

	isl_tarjan_graph_free(g);

	return 0;
}

/* Apply Tarjan's algorithm to detect the strongly connected components
 * in the dependence graph.
 * Only consider the (conditional) validity dependences and clear "weak".
 */
static int detect_sccs(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	graph->weak = 0;
	return detect_ccs(ctx, graph, &node_follows_strong);
}

/* Apply Tarjan's algorithm to detect the (weakly) connected components
 * in the dependence graph.
 * Consider all dependences and set "weak".
 */
static int detect_wccs(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	graph->weak = 1;
	return detect_ccs(ctx, graph, &node_follows_weak);
}

static int cmp_scc(const void *a, const void *b, void *data)
{
	struct isl_sched_graph *graph = data;
	const int *i1 = a;
	const int *i2 = b;

	return graph->node[*i1].scc - graph->node[*i2].scc;
}

/* Sort the elements of graph->sorted according to the corresponding SCCs.
 */
static int sort_sccs(struct isl_sched_graph *graph)
{
	return isl_sort(graph->sorted, graph->n, sizeof(int), &cmp_scc, graph);
}

/* Given a dependence relation R from "node" to itself,
 * construct the set of coefficients of valid constraints for elements
 * in that dependence relation.
 * In particular, the result contains tuples of coefficients
 * c_0, c_n, c_x such that
 *
 *	c_0 + c_n n + c_x y - c_x x >= 0 for each (x,y) in R
 *
 * or, equivalently,
 *
 *	c_0 + c_n n + c_x d >= 0 for each d in delta R = { y - x | (x,y) in R }
 *
 * We choose here to compute the dual of delta R.
 * Alternatively, we could have computed the dual of R, resulting
 * in a set of tuples c_0, c_n, c_x, c_y, and then
 * plugged in (c_0, c_n, c_x, -c_x).
 *
 * If "node" has been compressed, then the dependence relation
 * is also compressed before the set of coefficients is computed.
 */
static __isl_give isl_basic_set *intra_coefficients(
	struct isl_sched_graph *graph, struct isl_sched_node *node,
	__isl_take isl_map *map)
{
	isl_set *delta;
	isl_map *key;
	isl_basic_set *coef;
	isl_maybe_isl_basic_set m;

	m = isl_map_to_basic_set_try_get(graph->intra_hmap, map);
	if (m.valid < 0 || m.valid) {
		isl_map_free(map);
		return m.value;
	}

	key = isl_map_copy(map);
	if (node->compressed) {
		map = isl_map_preimage_domain_multi_aff(map,
				    isl_multi_aff_copy(node->decompress));
		map = isl_map_preimage_range_multi_aff(map,
				    isl_multi_aff_copy(node->decompress));
	}
	delta = isl_set_remove_divs(isl_map_deltas(map));
	coef = isl_set_coefficients(delta);
	graph->intra_hmap = isl_map_to_basic_set_set(graph->intra_hmap, key,
					isl_basic_set_copy(coef));

	return coef;
}

/* Given a dependence relation R, construct the set of coefficients
 * of valid constraints for elements in that dependence relation.
 * In particular, the result contains tuples of coefficients
 * c_0, c_n, c_x, c_y such that
 *
 *	c_0 + c_n n + c_x x + c_y y >= 0 for each (x,y) in R
 *
 * If the source or destination nodes of "edge" have been compressed,
 * then the dependence relation is also compressed before
 * the set of coefficients is computed.
 */
static __isl_give isl_basic_set *inter_coefficients(
	struct isl_sched_graph *graph, struct isl_sched_edge *edge,
	__isl_take isl_map *map)
{
	isl_set *set;
	isl_map *key;
	isl_basic_set *coef;
	isl_maybe_isl_basic_set m;

	m = isl_map_to_basic_set_try_get(graph->inter_hmap, map);
	if (m.valid < 0 || m.valid) {
		isl_map_free(map);
		return m.value;
	}

	key = isl_map_copy(map);
	if (edge->src->compressed)
		map = isl_map_preimage_domain_multi_aff(map,
				    isl_multi_aff_copy(edge->src->decompress));
	if (edge->dst->compressed)
		map = isl_map_preimage_range_multi_aff(map,
				    isl_multi_aff_copy(edge->dst->decompress));
	set = isl_map_wrap(isl_map_remove_divs(map));
	coef = isl_set_coefficients(set);
	graph->inter_hmap = isl_map_to_basic_set_set(graph->inter_hmap, key,
					isl_basic_set_copy(coef));

	return coef;
}

/* Return the position of the coefficients of the variables in
 * the coefficients constraints "coef".
 *
 * The space of "coef" is of the form
 *
 *	{ coefficients[[cst, params] -> S] }
 *
 * Return the position of S.
 */
static int coef_var_offset(__isl_keep isl_basic_set *coef)
{
	int offset;
	isl_space *space;

	space = isl_space_unwrap(isl_basic_set_get_space(coef));
	offset = isl_space_dim(space, isl_dim_in);
	isl_space_free(space);

	return offset;
}

/* Return the offset of the coefficients of the variables of "node"
 * within the (I)LP.
 *
 * Within each node, the coefficients have the following order:
 *	- c_i_0
 *	- c_i_n (if parametric)
 *	- positive and negative parts of c_i_x
 */
static int node_var_coef_offset(struct isl_sched_node *node)
{
	return node->start + 1 + node->nparam;
}

/* Construct an isl_dim_map for mapping constraints on coefficients
 * for "node" to the corresponding positions in graph->lp.
 * "offset" is the offset of the coefficients for the variables
 * in the input constraints.
 * "s" is the sign of the mapping.
 *
 * The input constraints are given in terms of the coefficients (c_0, c_n, c_x).
 * The mapping produced by this function essentially plugs in
 * (0, 0, c_i_x^+ - c_i_x^-) if s = 1 and
 * (0, 0, -c_i_x^+ + c_i_x^-) if s = -1.
 * In graph->lp, the c_i_x^- appear before their c_i_x^+ counterpart.
 *
 * The caller can extend the mapping to also map the other coefficients
 * (and therefore not plug in 0).
 */
static __isl_give isl_dim_map *intra_dim_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *node,
	int offset, int s)
{
	int pos;
	unsigned total;
	isl_dim_map *dim_map;

	total = isl_basic_set_total_dim(graph->lp);
	pos = node_var_coef_offset(node);
	dim_map = isl_dim_map_alloc(ctx, total);
	isl_dim_map_range(dim_map, pos, 2, offset, 1, node->nvar, -s);
	isl_dim_map_range(dim_map, pos + 1, 2, offset, 1, node->nvar, s);

	return dim_map;
}

/* Construct an isl_dim_map for mapping constraints on coefficients
 * for "src" (node i) and "dst" (node j) to the corresponding positions
 * in graph->lp.
 * "offset" is the offset of the coefficients for the variables of "src"
 * in the input constraints.
 * "s" is the sign of the mapping.
 *
 * The input constraints are given in terms of the coefficients
 * (c_0, c_n, c_x, c_y).
 * The mapping produced by this function essentially plugs in
 * (c_j_0 - c_i_0, c_j_n - c_i_n,
 *  c_j_x^+ - c_j_x^-, -(c_i_x^+ - c_i_x^-)) if s = 1 and
 * (-c_j_0 + c_i_0, -c_j_n + c_i_n,
 *  - (c_j_x^+ - c_j_x^-), c_i_x^+ - c_i_x^-) if s = -1.
 * In graph->lp, the c_*^- appear before their c_*^+ counterpart.
 *
 * The caller can further extend the mapping.
 */
static __isl_give isl_dim_map *inter_dim_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *src,
	struct isl_sched_node *dst, int offset, int s)
{
	int pos;
	unsigned total;
	isl_dim_map *dim_map;

	total = isl_basic_set_total_dim(graph->lp);
	dim_map = isl_dim_map_alloc(ctx, total);

	isl_dim_map_range(dim_map, dst->start, 0, 0, 0, 1, s);
	isl_dim_map_range(dim_map, dst->start + 1, 1, 1, 1, dst->nparam, s);
	pos = node_var_coef_offset(dst);
	isl_dim_map_range(dim_map, pos, 2, offset + src->nvar, 1,
			  dst->nvar, -s);
	isl_dim_map_range(dim_map, pos + 1, 2, offset + src->nvar, 1,
			  dst->nvar, s);

	isl_dim_map_range(dim_map, src->start, 0, 0, 0, 1, -s);
	isl_dim_map_range(dim_map, src->start + 1, 1, 1, 1, src->nparam, -s);
	pos = node_var_coef_offset(src);
	isl_dim_map_range(dim_map, pos, 2, offset, 1, src->nvar, s);
	isl_dim_map_range(dim_map, pos + 1, 2, offset, 1, src->nvar, -s);

	return dim_map;
}

/* Add constraints to graph->lp that force validity for the given
 * dependence from a node i to itself.
 * That is, add constraints that enforce
 *
 *	(c_i_0 + c_i_n n + c_i_x y) - (c_i_0 + c_i_n n + c_i_x x)
 *	= c_i_x (y - x) >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x)
 * of valid constraints for (y - x) and then plug in (0, 0, c_i_x^+ - c_i_x^-),
 * where c_i_x = c_i_x^+ - c_i_x^-, with c_i_x^+ and c_i_x^- non-negative.
 * In graph->lp, the c_i_x^- appear before their c_i_x^+ counterpart.
 *
 * Actually, we do not construct constraints for the c_i_x themselves,
 * but for the coefficients of c_i_x written as a linear combination
 * of the columns in node->cmap.
 */
static isl_stat add_intra_validity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	int offset;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *node = edge->src;

	coef = intra_coefficients(graph, node, map);

	offset = coef_var_offset(coef);

	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
					    offset, isl_mat_copy(node->cmap));
	if (!coef)
		return isl_stat_error;

	dim_map = intra_dim_map(ctx, graph, node, offset, 1);
	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);

	return isl_stat_ok;
}

/* Add constraints to graph->lp that force validity for the given
 * dependence from node i to node j.
 * That is, add constraints that enforce
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x, c_y)
 * of valid constraints for R and then plug in
 * (c_j_0 - c_i_0, c_j_n - c_i_n, c_j_x^+ - c_j_x^- - (c_i_x^+ - c_i_x^-)),
 * where c_* = c_*^+ - c_*^-, with c_*^+ and c_*^- non-negative.
 * In graph->lp, the c_*^- appear before their c_*^+ counterpart.
 *
 * Actually, we do not construct constraints for the c_*_x themselves,
 * but for the coefficients of c_*_x written as a linear combination
 * of the columns in node->cmap.
 */
static isl_stat add_inter_validity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	int offset;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *src = edge->src;
	struct isl_sched_node *dst = edge->dst;

	coef = inter_coefficients(graph, edge, map);

	offset = coef_var_offset(coef);

	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
				offset, isl_mat_copy(src->cmap));
	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
				offset + src->nvar, isl_mat_copy(dst->cmap));
	if (!coef)
		return isl_stat_error;

	dim_map = inter_dim_map(ctx, graph, src, dst, offset, 1);

	edge->start = graph->lp->n_ineq;
	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);
	if (!graph->lp)
		return isl_stat_error;
	edge->end = graph->lp->n_ineq;

	return isl_stat_ok;
}

/* Add constraints to graph->lp that bound the dependence distance for the given
 * dependence from a node i to itself.
 * If s = 1, we add the constraint
 *
 *	c_i_x (y - x) <= m_0 + m_n n
 *
 * or
 *
 *	-c_i_x (y - x) + m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * If s = -1, we add the constraint
 *
 *	-c_i_x (y - x) <= m_0 + m_n n
 *
 * or
 *
 *	c_i_x (y - x) + m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x)
 * of valid constraints for (y - x) and then plug in (m_0, m_n, -s * c_i_x),
 * with each coefficient (except m_0) represented as a pair of non-negative
 * coefficients.
 *
 * Actually, we do not construct constraints for the c_i_x themselves,
 * but for the coefficients of c_i_x written as a linear combination
 * of the columns in node->cmap.
 *
 *
 * If "local" is set, then we add constraints
 *
 *	c_i_x (y - x) <= 0
 *
 * or
 *
 *	-c_i_x (y - x) <= 0
 *
 * instead, forcing the dependence distance to be (less than or) equal to 0.
 * That is, we plug in (0, 0, -s * c_i_x),
 * Note that dependences marked local are treated as validity constraints
 * by add_all_validity_constraints and therefore also have
 * their distances bounded by 0 from below.
 */
static isl_stat add_intra_proximity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, int s, int local)
{
	int offset;
	unsigned nparam;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *node = edge->src;

	coef = intra_coefficients(graph, node, map);

	offset = coef_var_offset(coef);

	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
					    offset, isl_mat_copy(node->cmap));
	if (!coef)
		return isl_stat_error;

	nparam = isl_space_dim(node->space, isl_dim_param);
	dim_map = intra_dim_map(ctx, graph, node, offset, -s);

	if (!local) {
		isl_dim_map_range(dim_map, 1, 0, 0, 0, 1, 1);
		isl_dim_map_range(dim_map, 4, 2, 1, 1, nparam, -1);
		isl_dim_map_range(dim_map, 5, 2, 1, 1, nparam, 1);
	}
	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);

	return isl_stat_ok;
}

/* Add constraints to graph->lp that bound the dependence distance for the given
 * dependence from node i to node j.
 * If s = 1, we add the constraint
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x)
 *		<= m_0 + m_n n
 *
 * or
 *
 *	-(c_j_0 + c_j_n n + c_j_x y) + (c_i_0 + c_i_n n + c_i_x x) +
 *		m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * If s = -1, we add the constraint
 *
 *	-((c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x))
 *		<= m_0 + m_n n
 *
 * or
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) +
 *		m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x, c_y)
 * of valid constraints for R and then plug in
 * (m_0 - s*c_j_0 + s*c_i_0, m_n - s*c_j_n + s*c_i_n,
 *  -s*c_j_x+s*c_i_x)
 * with each coefficient (except m_0, c_*_0 and c_*_n)
 * represented as a pair of non-negative coefficients.
 *
 * Actually, we do not construct constraints for the c_*_x themselves,
 * but for the coefficients of c_*_x written as a linear combination
 * of the columns in node->cmap.
 *
 *
 * If "local" is set, then we add constraints
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) <= 0
 *
 * or
 *
 *	-((c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x)) <= 0
 *
 * instead, forcing the dependence distance to be (less than or) equal to 0.
 * That is, we plug in
 * (-s*c_j_0 + s*c_i_0, -s*c_j_n + s*c_i_n, -s*c_j_x+s*c_i_x).
 * Note that dependences marked local are treated as validity constraints
 * by add_all_validity_constraints and therefore also have
 * their distances bounded by 0 from below.
 */
static isl_stat add_inter_proximity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, int s, int local)
{
	int offset;
	unsigned nparam;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *src = edge->src;
	struct isl_sched_node *dst = edge->dst;

	coef = inter_coefficients(graph, edge, map);

	offset = coef_var_offset(coef);

	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
				offset, isl_mat_copy(src->cmap));
	coef = isl_basic_set_transform_dims(coef, isl_dim_set,
				offset + src->nvar, isl_mat_copy(dst->cmap));
	if (!coef)
		return isl_stat_error;

	nparam = isl_space_dim(src->space, isl_dim_param);
	dim_map = inter_dim_map(ctx, graph, src, dst, offset, -s);

	if (!local) {
		isl_dim_map_range(dim_map, 1, 0, 0, 0, 1, 1);
		isl_dim_map_range(dim_map, 4, 2, 1, 1, nparam, -1);
		isl_dim_map_range(dim_map, 5, 2, 1, 1, nparam, 1);
	}

	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);

	return isl_stat_ok;
}

/* Add all validity constraints to graph->lp.
 *
 * An edge that is forced to be local needs to have its dependence
 * distances equal to zero.  We take care of bounding them by 0 from below
 * here.  add_all_proximity_constraints takes care of bounding them by 0
 * from above.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int add_all_validity_constraints(struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge= &graph->edge[i];
		int local;

		local = is_local(edge) ||
			(is_coincidence(edge) && use_coincidence);
		if (!is_validity(edge) && !local)
			continue;
		if (edge->src != edge->dst)
			continue;
		if (add_intra_validity_constraints(graph, edge) < 0)
			return -1;
	}

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int local;

		local = is_local(edge) ||
			(is_coincidence(edge) && use_coincidence);
		if (!is_validity(edge) && !local)
			continue;
		if (edge->src == edge->dst)
			continue;
		if (add_inter_validity_constraints(graph, edge) < 0)
			return -1;
	}

	return 0;
}

/* Add constraints to graph->lp that bound the dependence distance
 * for all dependence relations.
 * If a given proximity dependence is identical to a validity
 * dependence, then the dependence distance is already bounded
 * from below (by zero), so we only need to bound the distance
 * from above.  (This includes the case of "local" dependences
 * which are treated as validity dependence by add_all_validity_constraints.)
 * Otherwise, we need to bound the distance both from above and from below.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int add_all_proximity_constraints(struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge= &graph->edge[i];
		int local;

		local = is_local(edge) ||
			(is_coincidence(edge) && use_coincidence);
		if (!is_proximity(edge) && !local)
			continue;
		if (edge->src == edge->dst &&
		    add_intra_proximity_constraints(graph, edge, 1, local) < 0)
			return -1;
		if (edge->src != edge->dst &&
		    add_inter_proximity_constraints(graph, edge, 1, local) < 0)
			return -1;
		if (is_validity(edge) || local)
			continue;
		if (edge->src == edge->dst &&
		    add_intra_proximity_constraints(graph, edge, -1, 0) < 0)
			return -1;
		if (edge->src != edge->dst &&
		    add_inter_proximity_constraints(graph, edge, -1, 0) < 0)
			return -1;
	}

	return 0;
}

/* Compute a basis for the rows in the linear part of the schedule
 * and extend this basis to a full basis.  The remaining rows
 * can then be used to force linear independence from the rows
 * in the schedule.
 *
 * In particular, given the schedule rows S, we compute
 *
 *	S   = H Q
 *	S U = H
 *
 * with H the Hermite normal form of S.  That is, all but the
 * first rank columns of H are zero and so each row in S is
 * a linear combination of the first rank rows of Q.
 * The matrix Q is then transposed because we will write the
 * coefficients of the next schedule row as a column vector s
 * and express this s as a linear combination s = Q c of the
 * computed basis.
 * Similarly, the matrix U is transposed such that we can
 * compute the coefficients c = U s from a schedule row s.
 */
static int node_update_cmap(struct isl_sched_node *node)
{
	isl_mat *H, *U, *Q;
	int n_row = isl_mat_rows(node->sched);

	H = isl_mat_sub_alloc(node->sched, 0, n_row,
			      1 + node->nparam, node->nvar);

	H = isl_mat_left_hermite(H, 0, &U, &Q);
	isl_mat_free(node->cmap);
	isl_mat_free(node->cinv);
	isl_mat_free(node->ctrans);
	node->ctrans = isl_mat_copy(Q);
	node->cmap = isl_mat_transpose(Q);
	node->cinv = isl_mat_transpose(U);
	node->rank = isl_mat_initial_non_zero_cols(H);
	isl_mat_free(H);

	if (!node->cmap || !node->cinv || !node->ctrans || node->rank < 0)
		return -1;
	return 0;
}

/* Is "edge" marked as a validity or a conditional validity edge?
 */
static int is_any_validity(struct isl_sched_edge *edge)
{
	return is_validity(edge) || is_conditional_validity(edge);
}

/* How many times should we count the constraints in "edge"?
 *
 * If carry is set, then we are counting the number of
 * (validity or conditional validity) constraints that will be added
 * in setup_carry_lp and we count each edge exactly once.
 *
 * Otherwise, we count as follows
 * validity		-> 1 (>= 0)
 * validity+proximity	-> 2 (>= 0 and upper bound)
 * proximity		-> 2 (lower and upper bound)
 * local(+any)		-> 2 (>= 0 and <= 0)
 *
 * If an edge is only marked conditional_validity then it counts
 * as zero since it is only checked afterwards.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int edge_multiplicity(struct isl_sched_edge *edge, int carry,
	int use_coincidence)
{
	if (carry)
		return 1;
	if (is_proximity(edge) || is_local(edge))
		return 2;
	if (use_coincidence && is_coincidence(edge))
		return 2;
	if (is_validity(edge))
		return 1;
	return 0;
}

/* Count the number of equality and inequality constraints
 * that will be added for the given map.
 *
 * "use_coincidence" is set if we should take into account coincidence edges.
 */
static int count_map_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, __isl_take isl_map *map,
	int *n_eq, int *n_ineq, int carry, int use_coincidence)
{
	isl_basic_set *coef;
	int f = edge_multiplicity(edge, carry, use_coincidence);

	if (f == 0) {
		isl_map_free(map);
		return 0;
	}

	if (edge->src == edge->dst)
		coef = intra_coefficients(graph, edge->src, map);
	else
		coef = inter_coefficients(graph, edge, map);
	if (!coef)
		return -1;
	*n_eq += f * coef->n_eq;
	*n_ineq += f * coef->n_ineq;
	isl_basic_set_free(coef);

	return 0;
}

/* Count the number of equality and inequality constraints
 * that will be added to the main lp problem.
 * We count as follows
 * validity		-> 1 (>= 0)
 * validity+proximity	-> 2 (>= 0 and upper bound)
 * proximity		-> 2 (lower and upper bound)
 * local(+any)		-> 2 (>= 0 and <= 0)
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int count_constraints(struct isl_sched_graph *graph,
	int *n_eq, int *n_ineq, int use_coincidence)
{
	int i;

	*n_eq = *n_ineq = 0;
	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge= &graph->edge[i];
		isl_map *map = isl_map_copy(edge->map);

		if (count_map_constraints(graph, edge, map, n_eq, n_ineq,
					    0, use_coincidence) < 0)
			return -1;
	}

	return 0;
}

/* Count the number of constraints that will be added by
 * add_bound_constant_constraints to bound the values of the constant terms
 * and increment *n_eq and *n_ineq accordingly.
 *
 * In practice, add_bound_constant_constraints only adds inequalities.
 */
static isl_stat count_bound_constant_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, int *n_eq, int *n_ineq)
{
	if (isl_options_get_schedule_max_constant_term(ctx) == -1)
		return isl_stat_ok;

	*n_ineq += graph->n;

	return isl_stat_ok;
}

/* Add constraints to bound the values of the constant terms in the schedule,
 * if requested by the user.
 *
 * The maximal value of the constant terms is defined by the option
 * "schedule_max_constant_term".
 *
 * Within each node, the coefficients have the following order:
 *	- c_i_0
 *	- c_i_n (if parametric)
 *	- positive and negative parts of c_i_x
 */
static isl_stat add_bound_constant_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i, k;
	int max;
	int total;

	max = isl_options_get_schedule_max_constant_term(ctx);
	if (max == -1)
		return isl_stat_ok;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		isl_seq_clr(graph->lp->ineq[k], 1 +  total);
		isl_int_set_si(graph->lp->ineq[k][1 + node->start], -1);
		isl_int_set_si(graph->lp->ineq[k][0], max);
	}

	return isl_stat_ok;
}

/* Count the number of constraints that will be added by
 * add_bound_coefficient_constraints and increment *n_eq and *n_ineq
 * accordingly.
 *
 * In practice, add_bound_coefficient_constraints only adds inequalities.
 */
static int count_bound_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, int *n_eq, int *n_ineq)
{
	int i;

	if (isl_options_get_schedule_max_coefficient(ctx) == -1 &&
	    !isl_options_get_schedule_treat_coalescing(ctx))
		return 0;

	for (i = 0; i < graph->n; ++i)
		*n_ineq += graph->node[i].nparam + 2 * graph->node[i].nvar;

	return 0;
}

/* Add constraints to graph->lp that bound the values of
 * the parameter schedule coefficients of "node" to "max" and
 * the variable schedule coefficients to the corresponding entry
 * in node->max.
 * In either case, a negative value means that no bound needs to be imposed.
 *
 * For parameter coefficients, this amounts to adding a constraint
 *
 *	c_n <= max
 *
 * i.e.,
 *
 *	-c_n + max >= 0
 *
 * The variables coefficients are, however, not represented directly.
 * Instead, the variables coefficients c_x are written as a linear
 * combination c_x = cmap c_z of some other coefficients c_z,
 * which are in turn encoded as c_z = c_z^+ - c_z^-.
 * Let a_j be the elements of row i of node->cmap, then
 *
 *	-max_i <= c_x_i <= max_i
 *
 * is encoded as
 *
 *	-max_i <= \sum_j a_j (c_z_j^+ - c_z_j^-) <= max_i
 *
 * or
 *
 *	-\sum_j a_j (c_z_j^+ - c_z_j^-) + max_i >= 0
 *	\sum_j a_j (c_z_j^+ - c_z_j^-) + max_i >= 0
 */
static isl_stat node_add_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *node, int max)
{
	int i, j, k;
	int total;
	isl_vec *ineq;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);

	for (j = 0; j < node->nparam; ++j) {
		int dim;

		if (max < 0)
			continue;

		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		dim = 1 + node->start + 1 + j;
		isl_seq_clr(graph->lp->ineq[k], 1 + total);
		isl_int_set_si(graph->lp->ineq[k][dim], -1);
		isl_int_set_si(graph->lp->ineq[k][0], max);
	}

	ineq = isl_vec_alloc(ctx, 1 + total);
	ineq = isl_vec_clr(ineq);
	if (!ineq)
		return isl_stat_error;
	for (i = 0; i < node->nvar; ++i) {
		int pos = 1 + node_var_coef_offset(node);

		if (isl_int_is_neg(node->max->el[i]))
			continue;

		for (j = 0; j < node->nvar; ++j) {
			isl_int_set(ineq->el[pos + 2 * j],
					node->cmap->row[i][j]);
			isl_int_neg(ineq->el[pos + 2 * j + 1],
					node->cmap->row[i][j]);
		}
		isl_int_set(ineq->el[0], node->max->el[i]);

		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			goto error;
		isl_seq_cpy(graph->lp->ineq[k], ineq->el, 1 + total);

		isl_seq_neg(ineq->el + pos, ineq->el + pos, 2 * node->nvar);
		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			goto error;
		isl_seq_cpy(graph->lp->ineq[k], ineq->el, 1 + total);
	}
	isl_vec_free(ineq);

	return isl_stat_ok;
error:
	isl_vec_free(ineq);
	return isl_stat_error;
}

/* Add constraints that bound the values of the variable and parameter
 * coefficients of the schedule.
 *
 * The maximal value of the coefficients is defined by the option
 * 'schedule_max_coefficient' and the entries in node->max.
 * These latter entries are only set if either the schedule_max_coefficient
 * option or the schedule_treat_coalescing option is set.
 */
static isl_stat add_bound_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	int max;

	max = isl_options_get_schedule_max_coefficient(ctx);

	if (max == -1 && !isl_options_get_schedule_treat_coalescing(ctx))
		return isl_stat_ok;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		if (node_add_coefficient_constraints(ctx, graph, node, max) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the "n" values starting at "first".
 */
static isl_stat add_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos, int first, int n)
{
	int i, k;
	int total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 +  total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < n; ++i)
		isl_int_set_si(graph->lp->eq[k][1 + first + i], 1);

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the parameter coefficients of all nodes.
 *
 * Within each node, the coefficients have the following order:
 *	- c_i_0
 *	- c_i_n (if parametric)
 *	- positive and negative parts of c_i_x
 */
static isl_stat add_param_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos)
{
	int i, j, k;
	int total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 +  total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < graph->n; ++i) {
		int pos = 1 + graph->node[i].start + 1;

		for (j = 0; j < graph->node[i].nparam; ++j)
			isl_int_set_si(graph->lp->eq[k][pos + j], 1);
	}

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the variable coefficients of all nodes.
 *
 * Within each node, the coefficients have the following order:
 *	- c_i_0
 *	- c_i_n (if parametric)
 *	- positive and negative parts of c_i_x
 */
static isl_stat add_var_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos)
{
	int i, j, k;
	int total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 +  total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int pos = 1 + node_var_coef_offset(node);

		for (j = 0; j < 2 * node->nvar; ++j)
			isl_int_set_si(graph->lp->eq[k][pos + j], 1);
	}

	return isl_stat_ok;
}

/* Construct an ILP problem for finding schedule coefficients
 * that result in non-negative, but small dependence distances
 * over all dependences.
 * In particular, the dependence distances over proximity edges
 * are bounded by m_0 + m_n n and we compute schedule coefficients
 * with small values (preferably zero) of m_n and m_0.
 *
 * All variables of the ILP are non-negative.  The actual coefficients
 * may be negative, so each coefficient is represented as the difference
 * of two non-negative variables.  The negative part always appears
 * immediately before the positive part.
 * Other than that, the variables have the following order
 *
 *	- sum of positive and negative parts of m_n coefficients
 *	- m_0
 *	- sum of all c_n coefficients
 *		(unconstrained when computing non-parametric schedules)
 *	- sum of positive and negative parts of all c_x coefficients
 *	- positive and negative parts of m_n coefficients
 *	- for each node
 *		- c_i_0
 *		- c_i_n (if parametric)
 *		- positive and negative parts of c_i_x
 *
 * The c_i_x are not represented directly, but through the columns of
 * node->cmap.  That is, the computed values are for variable t_i_x
 * such that c_i_x = Q t_i_x with Q equal to node->cmap.
 *
 * The constraints are those from the edges plus two or three equalities
 * to express the sums.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static isl_stat setup_lp(isl_ctx *ctx, struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;
	unsigned nparam;
	unsigned total;
	isl_space *space;
	int parametric;
	int param_pos;
	int n_eq, n_ineq;

	parametric = ctx->opt->schedule_parametric;
	nparam = isl_space_dim(graph->node[0].space, isl_dim_param);
	param_pos = 4;
	total = param_pos + 2 * nparam;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[graph->sorted[i]];
		if (node_update_cmap(node) < 0)
			return isl_stat_error;
		node->start = total;
		total += 1 + node->nparam + 2 * node->nvar;
	}

	if (count_constraints(graph, &n_eq, &n_ineq, use_coincidence) < 0)
		return isl_stat_error;
	if (count_bound_constant_constraints(ctx, graph, &n_eq, &n_ineq) < 0)
		return isl_stat_error;
	if (count_bound_coefficient_constraints(ctx, graph, &n_eq, &n_ineq) < 0)
		return isl_stat_error;

	space = isl_space_set_alloc(ctx, 0, total);
	isl_basic_set_free(graph->lp);
	n_eq += 2 + parametric;

	graph->lp = isl_basic_set_alloc_space(space, 0, n_eq, n_ineq);

	if (add_sum_constraint(graph, 0, param_pos, 2 * nparam) < 0)
		return isl_stat_error;
	if (parametric && add_param_sum_constraint(graph, 2) < 0)
		return isl_stat_error;
	if (add_var_sum_constraint(graph, 3) < 0)
		return isl_stat_error;
	if (add_bound_constant_constraints(ctx, graph) < 0)
		return isl_stat_error;
	if (add_bound_coefficient_constraints(ctx, graph) < 0)
		return isl_stat_error;
	if (add_all_validity_constraints(graph, use_coincidence) < 0)
		return isl_stat_error;
	if (add_all_proximity_constraints(graph, use_coincidence) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Analyze the conflicting constraint found by
 * isl_tab_basic_set_non_trivial_lexmin.  If it corresponds to the validity
 * constraint of one of the edges between distinct nodes, living, moreover
 * in distinct SCCs, then record the source and sink SCC as this may
 * be a good place to cut between SCCs.
 */
static int check_conflict(int con, void *user)
{
	int i;
	struct isl_sched_graph *graph = user;

	if (graph->src_scc >= 0)
		return 0;

	con -= graph->lp->n_eq;

	if (con >= graph->lp->n_ineq)
		return 0;

	for (i = 0; i < graph->n_edge; ++i) {
		if (!is_validity(&graph->edge[i]))
			continue;
		if (graph->edge[i].src == graph->edge[i].dst)
			continue;
		if (graph->edge[i].src->scc == graph->edge[i].dst->scc)
			continue;
		if (graph->edge[i].start > con)
			continue;
		if (graph->edge[i].end <= con)
			continue;
		graph->src_scc = graph->edge[i].src->scc;
		graph->dst_scc = graph->edge[i].dst->scc;
	}

	return 0;
}

/* Check whether the next schedule row of the given node needs to be
 * non-trivial.  Lower-dimensional domains may have some trivial rows,
 * but as soon as the number of remaining required non-trivial rows
 * is as large as the number or remaining rows to be computed,
 * all remaining rows need to be non-trivial.
 */
static int needs_row(struct isl_sched_graph *graph, struct isl_sched_node *node)
{
	return node->nvar - node->rank >= graph->maxvar - graph->n_row;
}

/* Solve the ILP problem constructed in setup_lp.
 * For each node such that all the remaining rows of its schedule
 * need to be non-trivial, we construct a non-triviality region.
 * This region imposes that the next row is independent of previous rows.
 * In particular the coefficients c_i_x are represented by t_i_x
 * variables with c_i_x = Q t_i_x and Q a unimodular matrix such that
 * its first columns span the rows of the previously computed part
 * of the schedule.  The non-triviality region enforces that at least
 * one of the remaining components of t_i_x is non-zero, i.e.,
 * that the new schedule row depends on at least one of the remaining
 * columns of Q.
 */
static __isl_give isl_vec *solve_lp(struct isl_sched_graph *graph)
{
	int i;
	isl_vec *sol;
	isl_basic_set *lp;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int skip = node->rank;
		graph->region[i].pos = node_var_coef_offset(node) + 2 * skip;
		if (needs_row(graph, node))
			graph->region[i].len = 2 * (node->nvar - skip);
		else
			graph->region[i].len = 0;
	}
	lp = isl_basic_set_copy(graph->lp);
	sol = isl_tab_basic_set_non_trivial_lexmin(lp, 2, graph->n,
				       graph->region, &check_conflict, graph);
	return sol;
}

/* Extract the coefficients for the variables of "node" from "sol".
 *
 * Within each node, the coefficients have the following order:
 *	- c_i_0
 *	- c_i_n (if parametric)
 *	- positive and negative parts of c_i_x
 *
 * The c_i_x^- appear before their c_i_x^+ counterpart.
 *
 * Return c_i_x = c_i_x^+ - c_i_x^-
 */
static __isl_give isl_vec *extract_var_coef(struct isl_sched_node *node,
	__isl_keep isl_vec *sol)
{
	int i;
	int pos;
	isl_vec *csol;

	if (!sol)
		return NULL;
	csol = isl_vec_alloc(isl_vec_get_ctx(sol), node->nvar);
	if (!csol)
		return NULL;

	pos = 1 + node_var_coef_offset(node);
	for (i = 0; i < node->nvar; ++i)
		isl_int_sub(csol->el[i],
			    sol->el[pos + 2 * i + 1], sol->el[pos + 2 * i]);

	return csol;
}

/* Update the schedules of all nodes based on the given solution
 * of the LP problem.
 * The new row is added to the current band.
 * All possibly negative coefficients are encoded as a difference
 * of two non-negative variables, so we need to perform the subtraction
 * here.  Moreover, if use_cmap is set, then the solution does
 * not refer to the actual coefficients c_i_x, but instead to variables
 * t_i_x such that c_i_x = Q t_i_x and Q is equal to node->cmap.
 * In this case, we then also need to perform this multiplication
 * to obtain the values of c_i_x.
 *
 * If coincident is set, then the caller guarantees that the new
 * row satisfies the coincidence constraints.
 */
static int update_schedule(struct isl_sched_graph *graph,
	__isl_take isl_vec *sol, int use_cmap, int coincident)
{
	int i, j;
	isl_vec *csol = NULL;

	if (!sol)
		goto error;
	if (sol->size == 0)
		isl_die(sol->ctx, isl_error_internal,
			"no solution found", goto error);
	if (graph->n_total_row >= graph->max_row)
		isl_die(sol->ctx, isl_error_internal,
			"too many schedule rows", goto error);

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int pos = node->start;
		int row = isl_mat_rows(node->sched);

		isl_vec_free(csol);
		csol = extract_var_coef(node, sol);
		if (!csol)
			goto error;

		isl_map_free(node->sched_map);
		node->sched_map = NULL;
		node->sched = isl_mat_add_rows(node->sched, 1);
		if (!node->sched)
			goto error;
		for (j = 0; j < 1 + node->nparam; ++j)
			node->sched = isl_mat_set_element(node->sched,
						row, j, sol->el[1 + pos + j]);
		if (use_cmap)
			csol = isl_mat_vec_product(isl_mat_copy(node->cmap),
						   csol);
		if (!csol)
			goto error;
		for (j = 0; j < node->nvar; ++j)
			node->sched = isl_mat_set_element(node->sched,
					row, 1 + node->nparam + j, csol->el[j]);
		node->coincident[graph->n_total_row] = coincident;
	}
	isl_vec_free(sol);
	isl_vec_free(csol);

	graph->n_row++;
	graph->n_total_row++;

	return 0;
error:
	isl_vec_free(sol);
	isl_vec_free(csol);
	return -1;
}

/* Convert row "row" of node->sched into an isl_aff living in "ls"
 * and return this isl_aff.
 */
static __isl_give isl_aff *extract_schedule_row(__isl_take isl_local_space *ls,
	struct isl_sched_node *node, int row)
{
	int j;
	isl_int v;
	isl_aff *aff;

	isl_int_init(v);

	aff = isl_aff_zero_on_domain(ls);
	isl_mat_get_element(node->sched, row, 0, &v);
	aff = isl_aff_set_constant(aff, v);
	for (j = 0; j < node->nparam; ++j) {
		isl_mat_get_element(node->sched, row, 1 + j, &v);
		aff = isl_aff_set_coefficient(aff, isl_dim_param, j, v);
	}
	for (j = 0; j < node->nvar; ++j) {
		isl_mat_get_element(node->sched, row, 1 + node->nparam + j, &v);
		aff = isl_aff_set_coefficient(aff, isl_dim_in, j, v);
	}

	isl_int_clear(v);

	return aff;
}

/* Convert the "n" rows starting at "first" of node->sched into a multi_aff
 * and return this multi_aff.
 *
 * The result is defined over the uncompressed node domain.
 */
static __isl_give isl_multi_aff *node_extract_partial_schedule_multi_aff(
	struct isl_sched_node *node, int first, int n)
{
	int i;
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff;
	isl_multi_aff *ma;
	int nrow;

	if (!node)
		return NULL;
	nrow = isl_mat_rows(node->sched);
	if (node->compressed)
		space = isl_multi_aff_get_domain_space(node->decompress);
	else
		space = isl_space_copy(node->space);
	ls = isl_local_space_from_space(isl_space_copy(space));
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, n);
	ma = isl_multi_aff_zero(space);

	for (i = first; i < first + n; ++i) {
		aff = extract_schedule_row(isl_local_space_copy(ls), node, i);
		ma = isl_multi_aff_set_aff(ma, i - first, aff);
	}

	isl_local_space_free(ls);

	if (node->compressed)
		ma = isl_multi_aff_pullback_multi_aff(ma,
					isl_multi_aff_copy(node->compress));

	return ma;
}

/* Convert node->sched into a multi_aff and return this multi_aff.
 *
 * The result is defined over the uncompressed node domain.
 */
static __isl_give isl_multi_aff *node_extract_schedule_multi_aff(
	struct isl_sched_node *node)
{
	int nrow;

	nrow = isl_mat_rows(node->sched);
	return node_extract_partial_schedule_multi_aff(node, 0, nrow);
}

/* Convert node->sched into a map and return this map.
 *
 * The result is cached in node->sched_map, which needs to be released
 * whenever node->sched is updated.
 * It is defined over the uncompressed node domain.
 */
static __isl_give isl_map *node_extract_schedule(struct isl_sched_node *node)
{
	if (!node->sched_map) {
		isl_multi_aff *ma;

		ma = node_extract_schedule_multi_aff(node);
		node->sched_map = isl_map_from_multi_aff(ma);
	}

	return isl_map_copy(node->sched_map);
}

/* Construct a map that can be used to update a dependence relation
 * based on the current schedule.
 * That is, construct a map expressing that source and sink
 * are executed within the same iteration of the current schedule.
 * This map can then be intersected with the dependence relation.
 * This is not the most efficient way, but this shouldn't be a critical
 * operation.
 */
static __isl_give isl_map *specializer(struct isl_sched_node *src,
	struct isl_sched_node *dst)
{
	isl_map *src_sched, *dst_sched;

	src_sched = node_extract_schedule(src);
	dst_sched = node_extract_schedule(dst);
	return isl_map_apply_range(src_sched, isl_map_reverse(dst_sched));
}

/* Intersect the domains of the nested relations in domain and range
 * of "umap" with "map".
 */
static __isl_give isl_union_map *intersect_domains(
	__isl_take isl_union_map *umap, __isl_keep isl_map *map)
{
	isl_union_set *uset;

	umap = isl_union_map_zip(umap);
	uset = isl_union_set_from_set(isl_map_wrap(isl_map_copy(map)));
	umap = isl_union_map_intersect_domain(umap, uset);
	umap = isl_union_map_zip(umap);
	return umap;
}

/* Update the dependence relation of the given edge based
 * on the current schedule.
 * If the dependence is carried completely by the current schedule, then
 * it is removed from the edge_tables.  It is kept in the list of edges
 * as otherwise all edge_tables would have to be recomputed.
 */
static int update_edge(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	int empty;
	isl_map *id;

	id = specializer(edge->src, edge->dst);
	edge->map = isl_map_intersect(edge->map, isl_map_copy(id));
	if (!edge->map)
		goto error;

	if (edge->tagged_condition) {
		edge->tagged_condition =
			intersect_domains(edge->tagged_condition, id);
		if (!edge->tagged_condition)
			goto error;
	}
	if (edge->tagged_validity) {
		edge->tagged_validity =
			intersect_domains(edge->tagged_validity, id);
		if (!edge->tagged_validity)
			goto error;
	}

	empty = isl_map_plain_is_empty(edge->map);
	if (empty < 0)
		goto error;
	if (empty)
		graph_remove_edge(graph, edge);

	isl_map_free(id);
	return 0;
error:
	isl_map_free(id);
	return -1;
}

/* Does the domain of "umap" intersect "uset"?
 */
static int domain_intersects(__isl_keep isl_union_map *umap,
	__isl_keep isl_union_set *uset)
{
	int empty;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(uset));
	empty = isl_union_map_is_empty(umap);
	isl_union_map_free(umap);

	return empty < 0 ? -1 : !empty;
}

/* Does the range of "umap" intersect "uset"?
 */
static int range_intersects(__isl_keep isl_union_map *umap,
	__isl_keep isl_union_set *uset)
{
	int empty;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_range(umap, isl_union_set_copy(uset));
	empty = isl_union_map_is_empty(umap);
	isl_union_map_free(umap);

	return empty < 0 ? -1 : !empty;
}

/* Are the condition dependences of "edge" local with respect to
 * the current schedule?
 *
 * That is, are domain and range of the condition dependences mapped
 * to the same point?
 *
 * In other words, is the condition false?
 */
static int is_condition_false(struct isl_sched_edge *edge)
{
	isl_union_map *umap;
	isl_map *map, *sched, *test;
	int empty, local;

	empty = isl_union_map_is_empty(edge->tagged_condition);
	if (empty < 0 || empty)
		return empty;

	umap = isl_union_map_copy(edge->tagged_condition);
	umap = isl_union_map_zip(umap);
	umap = isl_union_set_unwrap(isl_union_map_domain(umap));
	map = isl_map_from_union_map(umap);

	sched = node_extract_schedule(edge->src);
	map = isl_map_apply_domain(map, sched);
	sched = node_extract_schedule(edge->dst);
	map = isl_map_apply_range(map, sched);

	test = isl_map_identity(isl_map_get_space(map));
	local = isl_map_is_subset(map, test);
	isl_map_free(map);
	isl_map_free(test);

	return local;
}

/* For each conditional validity constraint that is adjacent
 * to a condition with domain in condition_source or range in condition_sink,
 * turn it into an unconditional validity constraint.
 */
static int unconditionalize_adjacent_validity(struct isl_sched_graph *graph,
	__isl_take isl_union_set *condition_source,
	__isl_take isl_union_set *condition_sink)
{
	int i;

	condition_source = isl_union_set_coalesce(condition_source);
	condition_sink = isl_union_set_coalesce(condition_sink);

	for (i = 0; i < graph->n_edge; ++i) {
		int adjacent;
		isl_union_map *validity;

		if (!is_conditional_validity(&graph->edge[i]))
			continue;
		if (is_validity(&graph->edge[i]))
			continue;

		validity = graph->edge[i].tagged_validity;
		adjacent = domain_intersects(validity, condition_sink);
		if (adjacent >= 0 && !adjacent)
			adjacent = range_intersects(validity, condition_source);
		if (adjacent < 0)
			goto error;
		if (!adjacent)
			continue;

		set_validity(&graph->edge[i]);
	}

	isl_union_set_free(condition_source);
	isl_union_set_free(condition_sink);
	return 0;
error:
	isl_union_set_free(condition_source);
	isl_union_set_free(condition_sink);
	return -1;
}

/* Update the dependence relations of all edges based on the current schedule
 * and enforce conditional validity constraints that are adjacent
 * to satisfied condition constraints.
 *
 * First check if any of the condition constraints are satisfied
 * (i.e., not local to the outer schedule) and keep track of
 * their domain and range.
 * Then update all dependence relations (which removes the non-local
 * constraints).
 * Finally, if any condition constraints turned out to be satisfied,
 * then turn all adjacent conditional validity constraints into
 * unconditional validity constraints.
 */
static int update_edges(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;
	int any = 0;
	isl_union_set *source, *sink;

	source = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	sink = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	for (i = 0; i < graph->n_edge; ++i) {
		int local;
		isl_union_set *uset;
		isl_union_map *umap;

		if (!is_condition(&graph->edge[i]))
			continue;
		if (is_local(&graph->edge[i]))
			continue;
		local = is_condition_false(&graph->edge[i]);
		if (local < 0)
			goto error;
		if (local)
			continue;

		any = 1;

		umap = isl_union_map_copy(graph->edge[i].tagged_condition);
		uset = isl_union_map_domain(umap);
		source = isl_union_set_union(source, uset);

		umap = isl_union_map_copy(graph->edge[i].tagged_condition);
		uset = isl_union_map_range(umap);
		sink = isl_union_set_union(sink, uset);
	}

	for (i = graph->n_edge - 1; i >= 0; --i) {
		if (update_edge(graph, &graph->edge[i]) < 0)
			goto error;
	}

	if (any)
		return unconditionalize_adjacent_validity(graph, source, sink);

	isl_union_set_free(source);
	isl_union_set_free(sink);
	return 0;
error:
	isl_union_set_free(source);
	isl_union_set_free(sink);
	return -1;
}

static void next_band(struct isl_sched_graph *graph)
{
	graph->band_start = graph->n_total_row;
}

/* Return the union of the universe domains of the nodes in "graph"
 * that satisfy "pred".
 */
static __isl_give isl_union_set *isl_sched_graph_domain(isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*pred)(struct isl_sched_node *node, int data), int data)
{
	int i;
	isl_set *set;
	isl_union_set *dom;

	for (i = 0; i < graph->n; ++i)
		if (pred(&graph->node[i], data))
			break;

	if (i >= graph->n)
		isl_die(ctx, isl_error_internal,
			"empty component", return NULL);

	set = isl_set_universe(isl_space_copy(graph->node[i].space));
	dom = isl_union_set_from_set(set);

	for (i = i + 1; i < graph->n; ++i) {
		if (!pred(&graph->node[i], data))
			continue;
		set = isl_set_universe(isl_space_copy(graph->node[i].space));
		dom = isl_union_set_union(dom, isl_union_set_from_set(set));
	}

	return dom;
}

/* Return a list of unions of universe domains, where each element
 * in the list corresponds to an SCC (or WCC) indexed by node->scc.
 */
static __isl_give isl_union_set_list *extract_sccs(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(ctx, graph->scc);
	for (i = 0; i < graph->scc; ++i) {
		isl_union_set *dom;

		dom = isl_sched_graph_domain(ctx, graph, &node_scc_exactly, i);
		filters = isl_union_set_list_add(filters, dom);
	}

	return filters;
}

/* Return a list of two unions of universe domains, one for the SCCs up
 * to and including graph->src_scc and another for the other SCCs.
 */
static __isl_give isl_union_set_list *extract_split(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	isl_union_set *dom;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(ctx, 2);
	dom = isl_sched_graph_domain(ctx, graph,
					&node_scc_at_most, graph->src_scc);
	filters = isl_union_set_list_add(filters, dom);
	dom = isl_sched_graph_domain(ctx, graph,
					&node_scc_at_least, graph->src_scc + 1);
	filters = isl_union_set_list_add(filters, dom);

	return filters;
}

/* Copy nodes that satisfy node_pred from the src dependence graph
 * to the dst dependence graph.
 */
static int copy_nodes(struct isl_sched_graph *dst, struct isl_sched_graph *src,
	int (*node_pred)(struct isl_sched_node *node, int data), int data)
{
	int i;

	dst->n = 0;
	for (i = 0; i < src->n; ++i) {
		int j;

		if (!node_pred(&src->node[i], data))
			continue;

		j = dst->n;
		dst->node[j].space = isl_space_copy(src->node[i].space);
		dst->node[j].compressed = src->node[i].compressed;
		dst->node[j].hull = isl_set_copy(src->node[i].hull);
		dst->node[j].compress =
			isl_multi_aff_copy(src->node[i].compress);
		dst->node[j].decompress =
			isl_multi_aff_copy(src->node[i].decompress);
		dst->node[j].nvar = src->node[i].nvar;
		dst->node[j].nparam = src->node[i].nparam;
		dst->node[j].sched = isl_mat_copy(src->node[i].sched);
		dst->node[j].sched_map = isl_map_copy(src->node[i].sched_map);
		dst->node[j].coincident = src->node[i].coincident;
		dst->node[j].sizes = isl_multi_val_copy(src->node[i].sizes);
		dst->node[j].max = isl_vec_copy(src->node[i].max);
		dst->n++;

		if (!dst->node[j].space || !dst->node[j].sched)
			return -1;
		if (dst->node[j].compressed &&
		    (!dst->node[j].hull || !dst->node[j].compress ||
		     !dst->node[j].decompress))
			return -1;
	}

	return 0;
}

/* Copy non-empty edges that satisfy edge_pred from the src dependence graph
 * to the dst dependence graph.
 * If the source or destination node of the edge is not in the destination
 * graph, then it must be a backward proximity edge and it should simply
 * be ignored.
 */
static int copy_edges(isl_ctx *ctx, struct isl_sched_graph *dst,
	struct isl_sched_graph *src,
	int (*edge_pred)(struct isl_sched_edge *edge, int data), int data)
{
	int i;
	enum isl_edge_type t;

	dst->n_edge = 0;
	for (i = 0; i < src->n_edge; ++i) {
		struct isl_sched_edge *edge = &src->edge[i];
		isl_map *map;
		isl_union_map *tagged_condition;
		isl_union_map *tagged_validity;
		struct isl_sched_node *dst_src, *dst_dst;

		if (!edge_pred(edge, data))
			continue;

		if (isl_map_plain_is_empty(edge->map))
			continue;

		dst_src = graph_find_node(ctx, dst, edge->src->space);
		dst_dst = graph_find_node(ctx, dst, edge->dst->space);
		if (!dst_src || !dst_dst) {
			if (is_validity(edge) || is_conditional_validity(edge))
				isl_die(ctx, isl_error_internal,
					"backward (conditional) validity edge",
					return -1);
			continue;
		}

		map = isl_map_copy(edge->map);
		tagged_condition = isl_union_map_copy(edge->tagged_condition);
		tagged_validity = isl_union_map_copy(edge->tagged_validity);

		dst->edge[dst->n_edge].src = dst_src;
		dst->edge[dst->n_edge].dst = dst_dst;
		dst->edge[dst->n_edge].map = map;
		dst->edge[dst->n_edge].tagged_condition = tagged_condition;
		dst->edge[dst->n_edge].tagged_validity = tagged_validity;
		dst->edge[dst->n_edge].types = edge->types;
		dst->n_edge++;

		if (edge->tagged_condition && !tagged_condition)
			return -1;
		if (edge->tagged_validity && !tagged_validity)
			return -1;

		for (t = isl_edge_first; t <= isl_edge_last; ++t) {
			if (edge !=
			    graph_find_edge(src, t, edge->src, edge->dst))
				continue;
			if (graph_edge_table_add(ctx, dst, t,
					    &dst->edge[dst->n_edge - 1]) < 0)
				return -1;
		}
	}

	return 0;
}

/* Compute the maximal number of variables over all nodes.
 * This is the maximal number of linearly independent schedule
 * rows that we need to compute.
 * Just in case we end up in a part of the dependence graph
 * with only lower-dimensional domains, we make sure we will
 * compute the required amount of extra linearly independent rows.
 */
static int compute_maxvar(struct isl_sched_graph *graph)
{
	int i;

	graph->maxvar = 0;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int nvar;

		if (node_update_cmap(node) < 0)
			return -1;
		nvar = node->nvar + graph->n_row - node->rank;
		if (nvar > graph->maxvar)
			graph->maxvar = nvar;
	}

	return 0;
}

/* Extract the subgraph of "graph" that consists of the node satisfying
 * "node_pred" and the edges satisfying "edge_pred" and store
 * the result in "sub".
 */
static int extract_sub_graph(isl_ctx *ctx, struct isl_sched_graph *graph,
	int (*node_pred)(struct isl_sched_node *node, int data),
	int (*edge_pred)(struct isl_sched_edge *edge, int data),
	int data, struct isl_sched_graph *sub)
{
	int i, n = 0, n_edge = 0;
	int t;

	for (i = 0; i < graph->n; ++i)
		if (node_pred(&graph->node[i], data))
			++n;
	for (i = 0; i < graph->n_edge; ++i)
		if (edge_pred(&graph->edge[i], data))
			++n_edge;
	if (graph_alloc(ctx, sub, n, n_edge) < 0)
		return -1;
	if (copy_nodes(sub, graph, node_pred, data) < 0)
		return -1;
	if (graph_init_table(ctx, sub) < 0)
		return -1;
	for (t = 0; t <= isl_edge_last; ++t)
		sub->max_edge[t] = graph->max_edge[t];
	if (graph_init_edge_tables(ctx, sub) < 0)
		return -1;
	if (copy_edges(ctx, sub, graph, edge_pred, data) < 0)
		return -1;
	sub->n_row = graph->n_row;
	sub->max_row = graph->max_row;
	sub->n_total_row = graph->n_total_row;
	sub->band_start = graph->band_start;

	return 0;
}

static __isl_give isl_schedule_node *compute_schedule(isl_schedule_node *node,
	struct isl_sched_graph *graph);
static __isl_give isl_schedule_node *compute_schedule_wcc(
	isl_schedule_node *node, struct isl_sched_graph *graph);

/* Compute a schedule for a subgraph of "graph".  In particular, for
 * the graph composed of nodes that satisfy node_pred and edges that
 * that satisfy edge_pred.
 * If the subgraph is known to consist of a single component, then wcc should
 * be set and then we call compute_schedule_wcc on the constructed subgraph.
 * Otherwise, we call compute_schedule, which will check whether the subgraph
 * is connected.
 *
 * The schedule is inserted at "node" and the updated schedule node
 * is returned.
 */
static __isl_give isl_schedule_node *compute_sub_schedule(
	__isl_take isl_schedule_node *node, isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*node_pred)(struct isl_sched_node *node, int data),
	int (*edge_pred)(struct isl_sched_edge *edge, int data),
	int data, int wcc)
{
	struct isl_sched_graph split = { 0 };

	if (extract_sub_graph(ctx, graph, node_pred, edge_pred, data,
				&split) < 0)
		goto error;

	if (wcc)
		node = compute_schedule_wcc(node, &split);
	else
		node = compute_schedule(node, &split);

	graph_free(ctx, &split);
	return node;
error:
	graph_free(ctx, &split);
	return isl_schedule_node_free(node);
}

static int edge_scc_exactly(struct isl_sched_edge *edge, int scc)
{
	return edge->src->scc == scc && edge->dst->scc == scc;
}

static int edge_dst_scc_at_most(struct isl_sched_edge *edge, int scc)
{
	return edge->dst->scc <= scc;
}

static int edge_src_scc_at_least(struct isl_sched_edge *edge, int scc)
{
	return edge->src->scc >= scc;
}

/* Reset the current band by dropping all its schedule rows.
 */
static int reset_band(struct isl_sched_graph *graph)
{
	int i;
	int drop;

	drop = graph->n_total_row - graph->band_start;
	graph->n_total_row -= drop;
	graph->n_row -= drop;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		isl_map_free(node->sched_map);
		node->sched_map = NULL;

		node->sched = isl_mat_drop_rows(node->sched,
						graph->band_start, drop);

		if (!node->sched)
			return -1;
	}

	return 0;
}

/* Split the current graph into two parts and compute a schedule for each
 * part individually.  In particular, one part consists of all SCCs up
 * to and including graph->src_scc, while the other part contains the other
 * SCCs.  The split is enforced by a sequence node inserted at position "node"
 * in the schedule tree.  Return the updated schedule node.
 * If either of these two parts consists of a sequence, then it is spliced
 * into the sequence containing the two parts.
 *
 * The current band is reset. It would be possible to reuse
 * the previously computed rows as the first rows in the next
 * band, but recomputing them may result in better rows as we are looking
 * at a smaller part of the dependence graph.
 */
static __isl_give isl_schedule_node *compute_split_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	int is_seq;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;

	if (reset_band(graph) < 0)
		return isl_schedule_node_free(node);

	next_band(graph);

	ctx = isl_schedule_node_get_ctx(node);
	filters = extract_split(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);
	node = isl_schedule_node_child(node, 1);
	node = isl_schedule_node_child(node, 0);

	node = compute_sub_schedule(node, ctx, graph,
				&node_scc_at_least, &edge_src_scc_at_least,
				graph->src_scc + 1, 0);
	is_seq = isl_schedule_node_get_type(node) == isl_schedule_node_sequence;
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	if (is_seq)
		node = isl_schedule_node_sequence_splice_child(node, 1);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, 0);
	node = compute_sub_schedule(node, ctx, graph,
				&node_scc_at_most, &edge_dst_scc_at_most,
				graph->src_scc, 0);
	is_seq = isl_schedule_node_get_type(node) == isl_schedule_node_sequence;
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	if (is_seq)
		node = isl_schedule_node_sequence_splice_child(node, 0);

	return node;
}

/* Insert a band node at position "node" in the schedule tree corresponding
 * to the current band in "graph".  Mark the band node permutable
 * if "permutable" is set.
 * The partial schedules and the coincidence property are extracted
 * from the graph nodes.
 * Return the updated schedule node.
 */
static __isl_give isl_schedule_node *insert_current_band(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int permutable)
{
	int i;
	int start, end, n;
	isl_multi_aff *ma;
	isl_multi_pw_aff *mpa;
	isl_multi_union_pw_aff *mupa;

	if (!node)
		return NULL;

	if (graph->n < 1)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
			"graph should have at least one node",
			return isl_schedule_node_free(node));

	start = graph->band_start;
	end = graph->n_total_row;
	n = end - start;

	ma = node_extract_partial_schedule_multi_aff(&graph->node[0], start, n);
	mpa = isl_multi_pw_aff_from_multi_aff(ma);
	mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

	for (i = 1; i < graph->n; ++i) {
		isl_multi_union_pw_aff *mupa_i;

		ma = node_extract_partial_schedule_multi_aff(&graph->node[i],
								start, n);
		mpa = isl_multi_pw_aff_from_multi_aff(ma);
		mupa_i = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);
		mupa = isl_multi_union_pw_aff_union_add(mupa, mupa_i);
	}
	node = isl_schedule_node_insert_partial_schedule(node, mupa);

	for (i = 0; i < n; ++i)
		node = isl_schedule_node_band_member_set_coincident(node, i,
					graph->node[0].coincident[start + i]);
	node = isl_schedule_node_band_set_permutable(node, permutable);

	return node;
}

/* Update the dependence relations based on the current schedule,
 * add the current band to "node" and then continue with the computation
 * of the next band.
 * Return the updated schedule node.
 */
static __isl_give isl_schedule_node *compute_next_band(
	__isl_take isl_schedule_node *node,
	struct isl_sched_graph *graph, int permutable)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (update_edges(ctx, graph) < 0)
		return isl_schedule_node_free(node);
	node = insert_current_band(node, graph, permutable);
	next_band(graph);

	node = isl_schedule_node_child(node, 0);
	node = compute_schedule(node, graph);
	node = isl_schedule_node_parent(node);

	return node;
}

/* Add constraints to graph->lp that force the dependence "map" (which
 * is part of the dependence relation of "edge")
 * to be respected and attempt to carry it, where the edge is one from
 * a node j to itself.  "pos" is the sequence number of the given map.
 * That is, add constraints that enforce
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_j_0 + c_j_n n + c_j_x x)
 *	= c_j_x (y - x) >= e_i
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x)
 * of valid constraints for (y - x) and then plug in (-e_i, 0, c_j_x),
 * with each coefficient in c_j_x represented as a pair of non-negative
 * coefficients.
 */
static int add_intra_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, __isl_take isl_map *map, int pos)
{
	int offset;
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *node = edge->src;

	coef = intra_coefficients(graph, node, map);
	if (!coef)
		return -1;

	offset = coef_var_offset(coef);
	dim_map = intra_dim_map(ctx, graph, node, offset, 1);
	isl_dim_map_range(dim_map, 3 + pos, 0, 0, 0, 1, -1);
	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);

	return 0;
}

/* Add constraints to graph->lp that force the dependence "map" (which
 * is part of the dependence relation of "edge")
 * to be respected and attempt to carry it, where the edge is one from
 * node j to node k.  "pos" is the sequence number of the given map.
 * That is, add constraints that enforce
 *
 *	(c_k_0 + c_k_n n + c_k_x y) - (c_j_0 + c_j_n n + c_j_x x) >= e_i
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x)
 * of valid constraints for R and then plug in
 * (-e_i + c_k_0 - c_j_0, c_k_n - c_j_n, c_k_x - c_j_x)
 * with each coefficient (except e_i, c_*_0 and c_*_n)
 * represented as a pair of non-negative coefficients.
 */
static int add_inter_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, __isl_take isl_map *map, int pos)
{
	int offset;
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *src = edge->src;
	struct isl_sched_node *dst = edge->dst;

	coef = inter_coefficients(graph, edge, map);
	if (!coef)
		return -1;

	offset = coef_var_offset(coef);
	dim_map = inter_dim_map(ctx, graph, src, dst, offset, 1);
	isl_dim_map_range(dim_map, 3 + pos, 0, 0, 0, 1, -1);
	graph->lp = isl_basic_set_extend_constraints(graph->lp,
			coef->n_eq, coef->n_ineq);
	graph->lp = isl_basic_set_add_constraints_dim_map(graph->lp,
							   coef, dim_map);

	return 0;
}

/* Add constraints to graph->lp that force all (conditional) validity
 * dependences to be respected and attempt to carry them.
 */
static int add_all_constraints(struct isl_sched_graph *graph)
{
	int i, j;
	int pos;

	pos = 0;
	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge= &graph->edge[i];

		if (!is_any_validity(edge))
			continue;

		for (j = 0; j < edge->map->n; ++j) {
			isl_basic_map *bmap;
			isl_map *map;

			bmap = isl_basic_map_copy(edge->map->p[j]);
			map = isl_map_from_basic_map(bmap);

			if (edge->src == edge->dst &&
			    add_intra_constraints(graph, edge, map, pos) < 0)
				return -1;
			if (edge->src != edge->dst &&
			    add_inter_constraints(graph, edge, map, pos) < 0)
				return -1;
			++pos;
		}
	}

	return 0;
}

/* Count the number of equality and inequality constraints
 * that will be added to the carry_lp problem.
 * We count each edge exactly once.
 */
static int count_all_constraints(struct isl_sched_graph *graph,
	int *n_eq, int *n_ineq)
{
	int i, j;

	*n_eq = *n_ineq = 0;
	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge= &graph->edge[i];

		if (!is_any_validity(edge))
			continue;

		for (j = 0; j < edge->map->n; ++j) {
			isl_basic_map *bmap;
			isl_map *map;

			bmap = isl_basic_map_copy(edge->map->p[j]);
			map = isl_map_from_basic_map(bmap);

			if (count_map_constraints(graph, edge, map,
						  n_eq, n_ineq, 1, 0) < 0)
				    return -1;
		}
	}

	return 0;
}

/* Construct an LP problem for finding schedule coefficients
 * such that the schedule carries as many dependences as possible.
 * In particular, for each dependence i, we bound the dependence distance
 * from below by e_i, with 0 <= e_i <= 1 and then maximize the sum
 * of all e_i's.  Dependences with e_i = 0 in the solution are simply
 * respected, while those with e_i > 0 (in practice e_i = 1) are carried.
 * Note that if the dependence relation is a union of basic maps,
 * then we have to consider each basic map individually as it may only
 * be possible to carry the dependences expressed by some of those
 * basic maps and not all of them.
 * Below, we consider each of those basic maps as a separate "edge".
 *
 * All variables of the LP are non-negative.  The actual coefficients
 * may be negative, so each coefficient is represented as the difference
 * of two non-negative variables.  The negative part always appears
 * immediately before the positive part.
 * Other than that, the variables have the following order
 *
 *	- sum of (1 - e_i) over all edges
 *	- sum of all c_n coefficients
 *		(unconstrained when computing non-parametric schedules)
 *	- sum of positive and negative parts of all c_x coefficients
 *	- for each edge
 *		- e_i
 *	- for each node
 *		- c_i_0
 *		- c_i_n (if parametric)
 *		- positive and negative parts of c_i_x
 *
 * The constraints are those from the (validity) edges plus three equalities
 * to express the sums and n_edge inequalities to express e_i <= 1.
 */
static isl_stat setup_carry_lp(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;
	int k;
	isl_space *dim;
	unsigned total;
	int n_eq, n_ineq;
	int n_edge;

	n_edge = 0;
	for (i = 0; i < graph->n_edge; ++i)
		n_edge += graph->edge[i].map->n;

	total = 3 + n_edge;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[graph->sorted[i]];
		node->start = total;
		total += 1 + node->nparam + 2 * node->nvar;
	}

	if (count_all_constraints(graph, &n_eq, &n_ineq) < 0)
		return isl_stat_error;

	dim = isl_space_set_alloc(ctx, 0, total);
	isl_basic_set_free(graph->lp);
	n_eq += 3;
	n_ineq += n_edge;
	graph->lp = isl_basic_set_alloc_space(dim, 0, n_eq, n_ineq);
	graph->lp = isl_basic_set_set_rational(graph->lp);

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 +  total);
	isl_int_set_si(graph->lp->eq[k][0], -n_edge);
	isl_int_set_si(graph->lp->eq[k][1], 1);
	for (i = 0; i < n_edge; ++i)
		isl_int_set_si(graph->lp->eq[k][4 + i], 1);

	if (add_param_sum_constraint(graph, 1) < 0)
		return isl_stat_error;
	if (add_var_sum_constraint(graph, 2) < 0)
		return isl_stat_error;

	for (i = 0; i < n_edge; ++i) {
		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		isl_seq_clr(graph->lp->ineq[k], 1 +  total);
		isl_int_set_si(graph->lp->ineq[k][4 + i], -1);
		isl_int_set_si(graph->lp->ineq[k][0], 1);
	}

	if (add_all_constraints(graph) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

static __isl_give isl_schedule_node *compute_component_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int wcc);

/* Comparison function for sorting the statements based on
 * the corresponding value in "r".
 */
static int smaller_value(const void *a, const void *b, void *data)
{
	isl_vec *r = data;
	const int *i1 = a;
	const int *i2 = b;

	return isl_int_cmp(r->el[*i1], r->el[*i2]);
}

/* If the schedule_split_scaled option is set and if the linear
 * parts of the scheduling rows for all nodes in the graphs have
 * a non-trivial common divisor, then split off the remainder of the
 * constant term modulo this common divisor from the linear part.
 * Otherwise, insert a band node directly and continue with
 * the construction of the schedule.
 *
 * If a non-trivial common divisor is found, then
 * the linear part is reduced and the remainder is enforced
 * by a sequence node with the children placed in the order
 * of this remainder.
 * In particular, we assign an scc index based on the remainder and
 * then rely on compute_component_schedule to insert the sequence and
 * to continue the schedule construction on each part.
 */
static __isl_give isl_schedule_node *split_scaled(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	int i;
	int row;
	int scc;
	isl_ctx *ctx;
	isl_int gcd, gcd_i;
	isl_vec *r;
	int *order;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (!ctx->opt->schedule_split_scaled)
		return compute_next_band(node, graph, 0);
	if (graph->n <= 1)
		return compute_next_band(node, graph, 0);

	isl_int_init(gcd);
	isl_int_init(gcd_i);

	isl_int_set_si(gcd, 0);

	row = isl_mat_rows(graph->node[0].sched) - 1;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int cols = isl_mat_cols(node->sched);

		isl_seq_gcd(node->sched->row[row] + 1, cols - 1, &gcd_i);
		isl_int_gcd(gcd, gcd, gcd_i);
	}

	isl_int_clear(gcd_i);

	if (isl_int_cmp_si(gcd, 1) <= 0) {
		isl_int_clear(gcd);
		return compute_next_band(node, graph, 0);
	}

	r = isl_vec_alloc(ctx, graph->n);
	order = isl_calloc_array(ctx, int, graph->n);
	if (!r || !order)
		goto error;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		order[i] = i;
		isl_int_fdiv_r(r->el[i], node->sched->row[row][0], gcd);
		isl_int_fdiv_q(node->sched->row[row][0],
			       node->sched->row[row][0], gcd);
		isl_int_mul(node->sched->row[row][0],
			    node->sched->row[row][0], gcd);
		node->sched = isl_mat_scale_down_row(node->sched, row, gcd);
		if (!node->sched)
			goto error;
	}

	if (isl_sort(order, graph->n, sizeof(order[0]), &smaller_value, r) < 0)
		goto error;

	scc = 0;
	for (i = 0; i < graph->n; ++i) {
		if (i > 0 && isl_int_ne(r->el[order[i - 1]], r->el[order[i]]))
			++scc;
		graph->node[order[i]].scc = scc;
	}
	graph->scc = ++scc;
	graph->weak = 0;

	isl_int_clear(gcd);
	isl_vec_free(r);
	free(order);

	if (update_edges(ctx, graph) < 0)
		return isl_schedule_node_free(node);
	node = insert_current_band(node, graph, 0);
	next_band(graph);

	node = isl_schedule_node_child(node, 0);
	node = compute_component_schedule(node, graph, 0);
	node = isl_schedule_node_parent(node);

	return node;
error:
	isl_vec_free(r);
	free(order);
	isl_int_clear(gcd);
	return isl_schedule_node_free(node);
}

/* Is the schedule row "sol" trivial on node "node"?
 * That is, is the solution zero on the dimensions orthogonal to
 * the previously found solutions?
 * Return 1 if the solution is trivial, 0 if it is not and -1 on error.
 *
 * Each coefficient is represented as the difference between
 * two non-negative values in "sol".  "sol" has been computed
 * in terms of the original iterators (i.e., without use of cmap).
 * We construct the schedule row s and write it as a linear
 * combination of (linear combinations of) previously computed schedule rows.
 * s = Q c or c = U s.
 * If the final entries of c are all zero, then the solution is trivial.
 */
static int is_trivial(struct isl_sched_node *node, __isl_keep isl_vec *sol)
{
	int trivial;
	isl_vec *node_sol;

	if (!sol)
		return -1;
	if (node->nvar == node->rank)
		return 0;

	node_sol = extract_var_coef(node, sol);
	node_sol = isl_mat_vec_product(isl_mat_copy(node->cinv), node_sol);
	if (!node_sol)
		return -1;

	trivial = isl_seq_first_non_zero(node_sol->el + node->rank,
					node->nvar - node->rank) == -1;

	isl_vec_free(node_sol);

	return trivial;
}

/* Is the schedule row "sol" trivial on any node where it should
 * not be trivial?
 * "sol" has been computed in terms of the original iterators
 * (i.e., without use of cmap).
 * Return 1 if any solution is trivial, 0 if they are not and -1 on error.
 */
static int is_any_trivial(struct isl_sched_graph *graph,
	__isl_keep isl_vec *sol)
{
	int i;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int trivial;

		if (!needs_row(graph, node))
			continue;
		trivial = is_trivial(node, sol);
		if (trivial < 0 || trivial)
			return trivial;
	}

	return 0;
}

/* Does the schedule represented by "sol" perform loop coalescing on "node"?
 * If so, return the position of the coalesced dimension.
 * Otherwise, return node->nvar or -1 on error.
 *
 * In particular, look for pairs of coefficients c_i and c_j such that
 * |c_j/c_i| >= size_i, i.e., |c_j| >= |c_i * size_i|.
 * If any such pair is found, then return i.
 * If size_i is infinity, then no check on c_i needs to be performed.
 */
static int find_node_coalescing(struct isl_sched_node *node,
	__isl_keep isl_vec *sol)
{
	int i, j;
	isl_int max;
	isl_vec *csol;

	if (node->nvar <= 1)
		return node->nvar;

	csol = extract_var_coef(node, sol);
	if (!csol)
		return -1;
	isl_int_init(max);
	for (i = 0; i < node->nvar; ++i) {
		isl_val *v;

		if (isl_int_is_zero(csol->el[i]))
			continue;
		v = isl_multi_val_get_val(node->sizes, i);
		if (!v)
			goto error;
		if (!isl_val_is_int(v)) {
			isl_val_free(v);
			continue;
		}
		isl_int_mul(max, v->n, csol->el[i]);
		isl_val_free(v);

		for (j = 0; j < node->nvar; ++j) {
			if (j == i)
				continue;
			if (isl_int_abs_ge(csol->el[j], max))
				break;
		}
		if (j < node->nvar)
			break;
	}

	isl_int_clear(max);
	isl_vec_free(csol);
	return i;
error:
	isl_int_clear(max);
	isl_vec_free(csol);
	return -1;
}

/* Force the schedule coefficient at position "pos" of "node" to be zero
 * in "tl".
 * The coefficient is encoded as the difference between two non-negative
 * variables.  Force these two variables to have the same value.
 */
static __isl_give isl_tab_lexmin *zero_out_node_coef(
	__isl_take isl_tab_lexmin *tl, struct isl_sched_node *node, int pos)
{
	int dim;
	isl_ctx *ctx;
	isl_vec *eq;

	ctx = isl_space_get_ctx(node->space);
	dim = isl_tab_lexmin_dim(tl);
	if (dim < 0)
		return isl_tab_lexmin_free(tl);
	eq = isl_vec_alloc(ctx, 1 + dim);
	eq = isl_vec_clr(eq);
	if (!eq)
		return isl_tab_lexmin_free(tl);

	pos = 1 + node_var_coef_offset(node) + 2 * pos;
	isl_int_set_si(eq->el[pos], 1);
	isl_int_set_si(eq->el[pos + 1], -1);
	tl = isl_tab_lexmin_add_eq(tl, eq->el);
	isl_vec_free(eq);

	return tl;
}

/* Return the lexicographically smallest rational point in the basic set
 * from which "tl" was constructed, double checking that this input set
 * was not empty.
 */
static __isl_give isl_vec *non_empty_solution(__isl_keep isl_tab_lexmin *tl)
{
	isl_vec *sol;

	sol = isl_tab_lexmin_get_solution(tl);
	if (!sol)
		return NULL;
	if (sol->size == 0)
		isl_die(isl_vec_get_ctx(sol), isl_error_internal,
			"error in schedule construction",
			return isl_vec_free(sol));
	return sol;
}

/* Does the solution "sol" of the LP problem constructed by setup_carry_lp
 * carry any of the "n_edge" groups of dependences?
 * The value in the first position is the sum of (1 - e_i) over all "n_edge"
 * edges, with 0 <= e_i <= 1 equal to 1 when the dependences represented
 * by the edge are carried by the solution.
 * If the sum of the (1 - e_i) is smaller than "n_edge" then at least
 * one of those is carried.
 *
 * Note that despite the fact that the problem is solved using a rational
 * solver, the solution is guaranteed to be integral.
 * Specifically, the dependence distance lower bounds e_i (and therefore
 * also their sum) are integers.  See Lemma 5 of [1].
 *
 * Any potential denominator of the sum is cleared by this function.
 * The denominator is not relevant for any of the other elements
 * in the solution.
 *
 * [1] P. Feautrier, Some Efficient Solutions to the Affine Scheduling
 *     Problem, Part II: Multi-Dimensional Time.
 *     In Intl. Journal of Parallel Programming, 1992.
 */
static int carries_dependences(__isl_keep isl_vec *sol, int n_edge)
{
	isl_int_divexact(sol->el[1], sol->el[1], sol->el[0]);
	isl_int_set_si(sol->el[0], 1);
	return isl_int_cmp_si(sol->el[1], n_edge) < 0;
}

/* Return the lexicographically smallest rational point in "lp",
 * assuming that all variables are non-negative and performing some
 * additional sanity checks.
 * In particular, "lp" should not be empty by construction.
 * Double check that this is the case.
 * Also, check that dependences are carried for at least one of
 * the "n_edge" edges.
 *
 * If the computed schedule performs loop coalescing on a given node,
 * i.e., if it is of the form
 *
 *	c_i i + c_j j + ...
 *
 * with |c_j/c_i| >= size_i, then force the coefficient c_i to be zero
 * to cut out this solution.  Repeat this process until no more loop
 * coalescing occurs or until no more dependences can be carried.
 * In the latter case, revert to the previously computed solution.
 */
static __isl_give isl_vec *non_neg_lexmin(struct isl_sched_graph *graph,
	__isl_take isl_basic_set *lp, int n_edge)
{
	int i, pos;
	isl_ctx *ctx;
	isl_tab_lexmin *tl;
	isl_vec *sol, *prev = NULL;
	int treat_coalescing;

	if (!lp)
		return NULL;
	ctx = isl_basic_set_get_ctx(lp);
	treat_coalescing = isl_options_get_schedule_treat_coalescing(ctx);
	tl = isl_tab_lexmin_from_basic_set(lp);

	do {
		sol = non_empty_solution(tl);
		if (!sol)
			goto error;

		if (!carries_dependences(sol, n_edge)) {
			if (!prev)
				isl_die(ctx, isl_error_unknown,
					"unable to carry dependences",
					goto error);
			isl_vec_free(sol);
			sol = prev;
			break;
		}
		prev = isl_vec_free(prev);
		if (!treat_coalescing)
			break;
		for (i = 0; i < graph->n; ++i) {
			struct isl_sched_node *node = &graph->node[i];

			pos = find_node_coalescing(node, sol);
			if (pos < 0)
				goto error;
			if (pos < node->nvar)
				break;
		}
		if (i < graph->n) {
			prev = sol;
			tl = zero_out_node_coef(tl, &graph->node[i], pos);
		}
	} while (i < graph->n);

	isl_tab_lexmin_free(tl);

	return sol;
error:
	isl_tab_lexmin_free(tl);
	isl_vec_free(prev);
	isl_vec_free(sol);
	return NULL;
}

/* Construct a schedule row for each node such that as many dependences
 * as possible are carried and then continue with the next band.
 *
 * If the computed schedule row turns out to be trivial on one or
 * more nodes where it should not be trivial, then we throw it away
 * and try again on each component separately.
 *
 * If there is only one component, then we accept the schedule row anyway,
 * but we do not consider it as a complete row and therefore do not
 * increment graph->n_row.  Note that the ranks of the nodes that
 * do get a non-trivial schedule part will get updated regardless and
 * graph->maxvar is computed based on these ranks.  The test for
 * whether more schedule rows are required in compute_schedule_wcc
 * is therefore not affected.
 *
 * Insert a band corresponding to the schedule row at position "node"
 * of the schedule tree and continue with the construction of the schedule.
 * This insertion and the continued construction is performed by split_scaled
 * after optionally checking for non-trivial common divisors.
 */
static __isl_give isl_schedule_node *carry_dependences(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	int i;
	int n_edge;
	int trivial;
	isl_ctx *ctx;
	isl_vec *sol;
	isl_basic_set *lp;

	if (!node)
		return NULL;

	n_edge = 0;
	for (i = 0; i < graph->n_edge; ++i)
		n_edge += graph->edge[i].map->n;

	ctx = isl_schedule_node_get_ctx(node);
	if (setup_carry_lp(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	lp = isl_basic_set_copy(graph->lp);
	sol = non_neg_lexmin(graph, lp, n_edge);
	if (!sol)
		return isl_schedule_node_free(node);

	trivial = is_any_trivial(graph, sol);
	if (trivial < 0) {
		sol = isl_vec_free(sol);
	} else if (trivial && graph->scc > 1) {
		isl_vec_free(sol);
		return compute_component_schedule(node, graph, 1);
	}

	if (update_schedule(graph, sol, 0, 0) < 0)
		return isl_schedule_node_free(node);
	if (trivial)
		graph->n_row--;

	return split_scaled(node, graph);
}

/* Topologically sort statements mapped to the same schedule iteration
 * and add insert a sequence node in front of "node"
 * corresponding to this order.
 * If "initialized" is set, then it may be assumed that compute_maxvar
 * has been called on the current band.  Otherwise, call
 * compute_maxvar if and before carry_dependences gets called.
 *
 * If it turns out to be impossible to sort the statements apart,
 * because different dependences impose different orderings
 * on the statements, then we extend the schedule such that
 * it carries at least one more dependence.
 */
static __isl_give isl_schedule_node *sort_statements(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int initialized)
{
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (graph->n < 1)
		isl_die(ctx, isl_error_internal,
			"graph should have at least one node",
			return isl_schedule_node_free(node));

	if (graph->n == 1)
		return node;

	if (update_edges(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	if (graph->n_edge == 0)
		return node;

	if (detect_sccs(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	next_band(graph);
	if (graph->scc < graph->n) {
		if (!initialized && compute_maxvar(graph) < 0)
			return isl_schedule_node_free(node);
		return carry_dependences(node, graph);
	}

	filters = extract_sccs(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);

	return node;
}

/* Are there any (non-empty) (conditional) validity edges in the graph?
 */
static int has_validity_edges(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		int empty;

		empty = isl_map_plain_is_empty(graph->edge[i].map);
		if (empty < 0)
			return -1;
		if (empty)
			continue;
		if (is_any_validity(&graph->edge[i]))
			return 1;
	}

	return 0;
}

/* Should we apply a Feautrier step?
 * That is, did the user request the Feautrier algorithm and are
 * there any validity dependences (left)?
 */
static int need_feautrier_step(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	if (ctx->opt->schedule_algorithm != ISL_SCHEDULE_ALGORITHM_FEAUTRIER)
		return 0;

	return has_validity_edges(graph);
}

/* Compute a schedule for a connected dependence graph using Feautrier's
 * multi-dimensional scheduling algorithm and return the updated schedule node.
 *
 * The original algorithm is described in [1].
 * The main idea is to minimize the number of scheduling dimensions, by
 * trying to satisfy as many dependences as possible per scheduling dimension.
 *
 * [1] P. Feautrier, Some Efficient Solutions to the Affine Scheduling
 *     Problem, Part II: Multi-Dimensional Time.
 *     In Intl. Journal of Parallel Programming, 1992.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc_feautrier(
	isl_schedule_node *node, struct isl_sched_graph *graph)
{
	return carry_dependences(node, graph);
}

/* Turn off the "local" bit on all (condition) edges.
 */
static void clear_local_edges(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i)
		if (is_condition(&graph->edge[i]))
			clear_local(&graph->edge[i]);
}

/* Does "graph" have both condition and conditional validity edges?
 */
static int need_condition_check(struct isl_sched_graph *graph)
{
	int i;
	int any_condition = 0;
	int any_conditional_validity = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		if (is_condition(&graph->edge[i]))
			any_condition = 1;
		if (is_conditional_validity(&graph->edge[i]))
			any_conditional_validity = 1;
	}

	return any_condition && any_conditional_validity;
}

/* Does "graph" contain any coincidence edge?
 */
static int has_any_coincidence(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i)
		if (is_coincidence(&graph->edge[i]))
			return 1;

	return 0;
}

/* Extract the final schedule row as a map with the iteration domain
 * of "node" as domain.
 */
static __isl_give isl_map *final_row(struct isl_sched_node *node)
{
	isl_local_space *ls;
	isl_aff *aff;
	int row;

	row = isl_mat_rows(node->sched) - 1;
	ls = isl_local_space_from_space(isl_space_copy(node->space));
	aff = extract_schedule_row(ls, node, row);
	return isl_map_from_aff(aff);
}

/* Is the conditional validity dependence in the edge with index "edge_index"
 * violated by the latest (i.e., final) row of the schedule?
 * That is, is i scheduled after j
 * for any conditional validity dependence i -> j?
 */
static int is_violated(struct isl_sched_graph *graph, int edge_index)
{
	isl_map *src_sched, *dst_sched, *map;
	struct isl_sched_edge *edge = &graph->edge[edge_index];
	int empty;

	src_sched = final_row(edge->src);
	dst_sched = final_row(edge->dst);
	map = isl_map_copy(edge->map);
	map = isl_map_apply_domain(map, src_sched);
	map = isl_map_apply_range(map, dst_sched);
	map = isl_map_order_gt(map, isl_dim_in, 0, isl_dim_out, 0);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	if (empty < 0)
		return -1;

	return !empty;
}

/* Does "graph" have any satisfied condition edges that
 * are adjacent to the conditional validity constraint with
 * domain "conditional_source" and range "conditional_sink"?
 *
 * A satisfied condition is one that is not local.
 * If a condition was forced to be local already (i.e., marked as local)
 * then there is no need to check if it is in fact local.
 *
 * Additionally, mark all adjacent condition edges found as local.
 */
static int has_adjacent_true_conditions(struct isl_sched_graph *graph,
	__isl_keep isl_union_set *conditional_source,
	__isl_keep isl_union_set *conditional_sink)
{
	int i;
	int any = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		int adjacent, local;
		isl_union_map *condition;

		if (!is_condition(&graph->edge[i]))
			continue;
		if (is_local(&graph->edge[i]))
			continue;

		condition = graph->edge[i].tagged_condition;
		adjacent = domain_intersects(condition, conditional_sink);
		if (adjacent >= 0 && !adjacent)
			adjacent = range_intersects(condition,
							conditional_source);
		if (adjacent < 0)
			return -1;
		if (!adjacent)
			continue;

		set_local(&graph->edge[i]);

		local = is_condition_false(&graph->edge[i]);
		if (local < 0)
			return -1;
		if (!local)
			any = 1;
	}

	return any;
}

/* Are there any violated conditional validity dependences with
 * adjacent condition dependences that are not local with respect
 * to the current schedule?
 * That is, is the conditional validity constraint violated?
 *
 * Additionally, mark all those adjacent condition dependences as local.
 * We also mark those adjacent condition dependences that were not marked
 * as local before, but just happened to be local already.  This ensures
 * that they remain local if the schedule is recomputed.
 *
 * We first collect domain and range of all violated conditional validity
 * dependences and then check if there are any adjacent non-local
 * condition dependences.
 */
static int has_violated_conditional_constraint(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	int any = 0;
	isl_union_set *source, *sink;

	source = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	sink = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	for (i = 0; i < graph->n_edge; ++i) {
		isl_union_set *uset;
		isl_union_map *umap;
		int violated;

		if (!is_conditional_validity(&graph->edge[i]))
			continue;

		violated = is_violated(graph, i);
		if (violated < 0)
			goto error;
		if (!violated)
			continue;

		any = 1;

		umap = isl_union_map_copy(graph->edge[i].tagged_validity);
		uset = isl_union_map_domain(umap);
		source = isl_union_set_union(source, uset);
		source = isl_union_set_coalesce(source);

		umap = isl_union_map_copy(graph->edge[i].tagged_validity);
		uset = isl_union_map_range(umap);
		sink = isl_union_set_union(sink, uset);
		sink = isl_union_set_coalesce(sink);
	}

	if (any)
		any = has_adjacent_true_conditions(graph, source, sink);

	isl_union_set_free(source);
	isl_union_set_free(sink);
	return any;
error:
	isl_union_set_free(source);
	isl_union_set_free(sink);
	return -1;
}

/* Examine the current band (the rows between graph->band_start and
 * graph->n_total_row), deciding whether to drop it or add it to "node"
 * and then continue with the computation of the next band, if any.
 * If "initialized" is set, then it may be assumed that compute_maxvar
 * has been called on the current band.  Otherwise, call
 * compute_maxvar if and before carry_dependences gets called.
 *
 * The caller keeps looking for a new row as long as
 * graph->n_row < graph->maxvar.  If the latest attempt to find
 * such a row failed (i.e., we still have graph->n_row < graph->maxvar),
 * then we either
 * - split between SCCs and start over (assuming we found an interesting
 *	pair of SCCs between which to split)
 * - continue with the next band (assuming the current band has at least
 *	one row)
 * - try to carry as many dependences as possible and continue with the next
 *	band
 * In each case, we first insert a band node in the schedule tree
 * if any rows have been computed.
 *
 * If the caller managed to complete the schedule, we insert a band node
 * (if any schedule rows were computed) and we finish off by topologically
 * sorting the statements based on the remaining dependences.
 */
static __isl_give isl_schedule_node *compute_schedule_finish_band(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int initialized)
{
	int insert;

	if (!node)
		return NULL;

	if (graph->n_row < graph->maxvar) {
		isl_ctx *ctx;
		int empty = graph->n_total_row == graph->band_start;

		ctx = isl_schedule_node_get_ctx(node);
		if (!ctx->opt->schedule_maximize_band_depth && !empty)
			return compute_next_band(node, graph, 1);
		if (graph->src_scc >= 0)
			return compute_split_schedule(node, graph);
		if (!empty)
			return compute_next_band(node, graph, 1);
		if (!initialized && compute_maxvar(graph) < 0)
			return isl_schedule_node_free(node);
		return carry_dependences(node, graph);
	}

	insert = graph->n_total_row > graph->band_start;
	if (insert) {
		node = insert_current_band(node, graph, 1);
		node = isl_schedule_node_child(node, 0);
	}
	node = sort_statements(node, graph, initialized);
	if (insert)
		node = isl_schedule_node_parent(node);

	return node;
}

/* Construct a band of schedule rows for a connected dependence graph.
 * The caller is responsible for determining the strongly connected
 * components and calling compute_maxvar first.
 *
 * We try to find a sequence of as many schedule rows as possible that result
 * in non-negative dependence distances (independent of the previous rows
 * in the sequence, i.e., such that the sequence is tilable), with as
 * many of the initial rows as possible satisfying the coincidence constraints.
 * The computation stops if we can't find any more rows or if we have found
 * all the rows we wanted to find.
 *
 * If ctx->opt->schedule_outer_coincidence is set, then we force the
 * outermost dimension to satisfy the coincidence constraints.  If this
 * turns out to be impossible, we fall back on the general scheme above
 * and try to carry as many dependences as possible.
 *
 * If "graph" contains both condition and conditional validity dependences,
 * then we need to check that that the conditional schedule constraint
 * is satisfied, i.e., there are no violated conditional validity dependences
 * that are adjacent to any non-local condition dependences.
 * If there are, then we mark all those adjacent condition dependences
 * as local and recompute the current band.  Those dependences that
 * are marked local will then be forced to be local.
 * The initial computation is performed with no dependences marked as local.
 * If we are lucky, then there will be no violated conditional validity
 * dependences adjacent to any non-local condition dependences.
 * Otherwise, we mark some additional condition dependences as local and
 * recompute.  We continue this process until there are no violations left or
 * until we are no longer able to compute a schedule.
 * Since there are only a finite number of dependences,
 * there will only be a finite number of iterations.
 */
static isl_stat compute_schedule_wcc_band(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int has_coincidence;
	int use_coincidence;
	int force_coincidence = 0;
	int check_conditional;

	if (sort_sccs(graph) < 0)
		return isl_stat_error;

	clear_local_edges(graph);
	check_conditional = need_condition_check(graph);
	has_coincidence = has_any_coincidence(graph);

	if (ctx->opt->schedule_outer_coincidence)
		force_coincidence = 1;

	use_coincidence = has_coincidence;
	while (graph->n_row < graph->maxvar) {
		isl_vec *sol;
		int violated;
		int coincident;

		graph->src_scc = -1;
		graph->dst_scc = -1;

		if (setup_lp(ctx, graph, use_coincidence) < 0)
			return isl_stat_error;
		sol = solve_lp(graph);
		if (!sol)
			return isl_stat_error;
		if (sol->size == 0) {
			int empty = graph->n_total_row == graph->band_start;

			isl_vec_free(sol);
			if (use_coincidence && (!force_coincidence || !empty)) {
				use_coincidence = 0;
				continue;
			}
			return isl_stat_ok;
		}
		coincident = !has_coincidence || use_coincidence;
		if (update_schedule(graph, sol, 1, coincident) < 0)
			return isl_stat_error;

		if (!check_conditional)
			continue;
		violated = has_violated_conditional_constraint(ctx, graph);
		if (violated < 0)
			return isl_stat_error;
		if (!violated)
			continue;
		if (reset_band(graph) < 0)
			return isl_stat_error;
		use_coincidence = has_coincidence;
	}

	return isl_stat_ok;
}

/* Compute a schedule for a connected dependence graph by considering
 * the graph as a whole and return the updated schedule node.
 *
 * The actual schedule rows of the current band are computed by
 * compute_schedule_wcc_band.  compute_schedule_finish_band takes
 * care of integrating the band into "node" and continuing
 * the computation.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc_whole(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (compute_schedule_wcc_band(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	return compute_schedule_finish_band(node, graph, 1);
}

/* Clustering information used by compute_schedule_wcc_clustering.
 *
 * "n" is the number of SCCs in the original dependence graph
 * "scc" is an array of "n" elements, each representing an SCC
 * of the original dependence graph.  All entries in the same cluster
 * have the same number of schedule rows.
 * "scc_cluster" maps each SCC index to the cluster to which it belongs,
 * where each cluster is represented by the index of the first SCC
 * in the cluster.  Initially, each SCC belongs to a cluster containing
 * only that SCC.
 *
 * "scc_in_merge" is used by merge_clusters_along_edge to keep
 * track of which SCCs need to be merged.
 *
 * "cluster" contains the merged clusters of SCCs after the clustering
 * has completed.
 *
 * "scc_node" is a temporary data structure used inside copy_partial.
 * For each SCC, it keeps track of the number of nodes in the SCC
 * that have already been copied.
 */
struct isl_clustering {
	int n;
	struct isl_sched_graph *scc;
	struct isl_sched_graph *cluster;
	int *scc_cluster;
	int *scc_node;
	int *scc_in_merge;
};

/* Initialize the clustering data structure "c" from "graph".
 *
 * In particular, allocate memory, extract the SCCs from "graph"
 * into c->scc, initialize scc_cluster and construct
 * a band of schedule rows for each SCC.
 * Within each SCC, there is only one SCC by definition.
 * Each SCC initially belongs to a cluster containing only that SCC.
 */
static isl_stat clustering_init(isl_ctx *ctx, struct isl_clustering *c,
	struct isl_sched_graph *graph)
{
	int i;

	c->n = graph->scc;
	c->scc = isl_calloc_array(ctx, struct isl_sched_graph, c->n);
	c->cluster = isl_calloc_array(ctx, struct isl_sched_graph, c->n);
	c->scc_cluster = isl_calloc_array(ctx, int, c->n);
	c->scc_node = isl_calloc_array(ctx, int, c->n);
	c->scc_in_merge = isl_calloc_array(ctx, int, c->n);
	if (!c->scc || !c->cluster ||
	    !c->scc_cluster || !c->scc_node || !c->scc_in_merge)
		return isl_stat_error;

	for (i = 0; i < c->n; ++i) {
		if (extract_sub_graph(ctx, graph, &node_scc_exactly,
					&edge_scc_exactly, i, &c->scc[i]) < 0)
			return isl_stat_error;
		c->scc[i].scc = 1;
		if (compute_maxvar(&c->scc[i]) < 0)
			return isl_stat_error;
		if (compute_schedule_wcc_band(ctx, &c->scc[i]) < 0)
			return isl_stat_error;
		c->scc_cluster[i] = i;
	}

	return isl_stat_ok;
}

/* Free all memory allocated for "c".
 */
static void clustering_free(isl_ctx *ctx, struct isl_clustering *c)
{
	int i;

	if (c->scc)
		for (i = 0; i < c->n; ++i)
			graph_free(ctx, &c->scc[i]);
	free(c->scc);
	if (c->cluster)
		for (i = 0; i < c->n; ++i)
			graph_free(ctx, &c->cluster[i]);
	free(c->cluster);
	free(c->scc_cluster);
	free(c->scc_node);
	free(c->scc_in_merge);
}

/* Should we refrain from merging the cluster in "graph" with
 * any other cluster?
 * In particular, is its current schedule band empty and incomplete.
 */
static int bad_cluster(struct isl_sched_graph *graph)
{
	return graph->n_row < graph->maxvar &&
		graph->n_total_row == graph->band_start;
}

/* Return the index of an edge in "graph" that can be used to merge
 * two clusters in "c".
 * Return graph->n_edge if no such edge can be found.
 * Return -1 on error.
 *
 * In particular, return a proximity edge between two clusters
 * that is not marked "no_merge" and such that neither of the
 * two clusters has an incomplete, empty band.
 *
 * If there are multiple such edges, then try and find the most
 * appropriate edge to use for merging.  In particular, pick the edge
 * with the greatest weight.  If there are multiple of those,
 * then pick one with the shortest distance between
 * the two cluster representatives.
 */
static int find_proximity(struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i, best = graph->n_edge, best_dist, best_weight;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int dist, weight;

		if (!is_proximity(edge))
			continue;
		if (edge->no_merge)
			continue;
		if (bad_cluster(&c->scc[edge->src->scc]) ||
		    bad_cluster(&c->scc[edge->dst->scc]))
			continue;
		dist = c->scc_cluster[edge->dst->scc] -
			c->scc_cluster[edge->src->scc];
		if (dist == 0)
			continue;
		weight = edge->weight;
		if (best < graph->n_edge) {
			if (best_weight > weight)
				continue;
			if (best_weight == weight && best_dist <= dist)
				continue;
		}
		best = i;
		best_dist = dist;
		best_weight = weight;
	}

	return best;
}

/* Internal data structure used in mark_merge_sccs.
 *
 * "graph" is the dependence graph in which a strongly connected
 * component is constructed.
 * "scc_cluster" maps each SCC index to the cluster to which it belongs.
 * "src" and "dst" are the indices of the nodes that are being merged.
 */
struct isl_mark_merge_sccs_data {
	struct isl_sched_graph *graph;
	int *scc_cluster;
	int src;
	int dst;
};

/* Check whether the cluster containing node "i" depends on the cluster
 * containing node "j".  If "i" and "j" belong to the same cluster,
 * then they are taken to depend on each other to ensure that
 * the resulting strongly connected component consists of complete
 * clusters.  Furthermore, if "i" and "j" are the two nodes that
 * are being merged, then they are taken to depend on each other as well.
 * Otherwise, check if there is a (conditional) validity dependence
 * from node[j] to node[i], forcing node[i] to follow node[j].
 */
static isl_bool cluster_follows(int i, int j, void *user)
{
	struct isl_mark_merge_sccs_data *data = user;
	struct isl_sched_graph *graph = data->graph;
	int *scc_cluster = data->scc_cluster;

	if (data->src == i && data->dst == j)
		return isl_bool_true;
	if (data->src == j && data->dst == i)
		return isl_bool_true;
	if (scc_cluster[graph->node[i].scc] == scc_cluster[graph->node[j].scc])
		return isl_bool_true;

	return graph_has_validity_edge(graph, &graph->node[j], &graph->node[i]);
}

/* Mark all SCCs that belong to either of the two clusters in "c"
 * connected by the edge in "graph" with index "edge", or to any
 * of the intermediate clusters.
 * The marking is recorded in c->scc_in_merge.
 *
 * The given edge has been selected for merging two clusters,
 * meaning that there is at least a proximity edge between the two nodes.
 * However, there may also be (indirect) validity dependences
 * between the two nodes.  When merging the two clusters, all clusters
 * containing one or more of the intermediate nodes along the
 * indirect validity dependences need to be merged in as well.
 *
 * First collect all such nodes by computing the strongly connected
 * component (SCC) containing the two nodes connected by the edge, where
 * the two nodes are considered to depend on each other to make
 * sure they end up in the same SCC.  Similarly, each node is considered
 * to depend on every other node in the same cluster to ensure
 * that the SCC consists of complete clusters.
 *
 * Then the original SCCs that contain any of these nodes are marked
 * in c->scc_in_merge.
 */
static isl_stat mark_merge_sccs(isl_ctx *ctx, struct isl_sched_graph *graph,
	int edge, struct isl_clustering *c)
{
	struct isl_mark_merge_sccs_data data;
	struct isl_tarjan_graph *g;
	int i;

	for (i = 0; i < c->n; ++i)
		c->scc_in_merge[i] = 0;

	data.graph = graph;
	data.scc_cluster = c->scc_cluster;
	data.src = graph->edge[edge].src - graph->node;
	data.dst = graph->edge[edge].dst - graph->node;

	g = isl_tarjan_graph_component(ctx, graph->n, data.dst,
					&cluster_follows, &data);
	if (!g)
		goto error;

	i = g->op;
	if (i < 3)
		isl_die(ctx, isl_error_internal,
			"expecting at least two nodes in component",
			goto error);
	if (g->order[--i] != -1)
		isl_die(ctx, isl_error_internal,
			"expecting end of component marker", goto error);

	for (--i; i >= 0 && g->order[i] != -1; --i) {
		int scc = graph->node[g->order[i]].scc;
		c->scc_in_merge[scc] = 1;
	}

	isl_tarjan_graph_free(g);
	return isl_stat_ok;
error:
	isl_tarjan_graph_free(g);
	return isl_stat_error;
}

/* Construct the identifier "cluster_i".
 */
static __isl_give isl_id *cluster_id(isl_ctx *ctx, int i)
{
	char name[40];

	snprintf(name, sizeof(name), "cluster_%d", i);
	return isl_id_alloc(ctx, name, NULL);
}

/* Construct the space of the cluster with index "i" containing
 * the strongly connected component "scc".
 *
 * In particular, construct a space called cluster_i with dimension equal
 * to the number of schedule rows in the current band of "scc".
 */
static __isl_give isl_space *cluster_space(struct isl_sched_graph *scc, int i)
{
	int nvar;
	isl_space *space;
	isl_id *id;

	nvar = scc->n_total_row - scc->band_start;
	space = isl_space_copy(scc->node[0].space);
	space = isl_space_params(space);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, nvar);
	id = cluster_id(isl_space_get_ctx(space), i);
	space = isl_space_set_tuple_id(space, isl_dim_set, id);

	return space;
}

/* Collect the domain of the graph for merging clusters.
 *
 * In particular, for each cluster with first SCC "i", construct
 * a set in the space called cluster_i with dimension equal
 * to the number of schedule rows in the current band of the cluster.
 */
static __isl_give isl_union_set *collect_domain(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i;
	isl_space *space;
	isl_union_set *domain;

	space = isl_space_params_alloc(ctx, 0);
	domain = isl_union_set_empty(space);

	for (i = 0; i < graph->scc; ++i) {
		isl_space *space;

		if (!c->scc_in_merge[i])
			continue;
		if (c->scc_cluster[i] != i)
			continue;
		space = cluster_space(&c->scc[i], i);
		domain = isl_union_set_add_set(domain, isl_set_universe(space));
	}

	return domain;
}

/* Construct a map from the original instances to the corresponding
 * cluster instance in the current bands of the clusters in "c".
 */
static __isl_give isl_union_map *collect_cluster_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i, j;
	isl_space *space;
	isl_union_map *cluster_map;

	space = isl_space_params_alloc(ctx, 0);
	cluster_map = isl_union_map_empty(space);
	for (i = 0; i < graph->scc; ++i) {
		int start, n;
		isl_id *id;

		if (!c->scc_in_merge[i])
			continue;

		id = cluster_id(ctx, c->scc_cluster[i]);
		start = c->scc[i].band_start;
		n = c->scc[i].n_total_row - start;
		for (j = 0; j < c->scc[i].n; ++j) {
			isl_multi_aff *ma;
			isl_map *map;
			struct isl_sched_node *node = &c->scc[i].node[j];

			ma = node_extract_partial_schedule_multi_aff(node,
								    start, n);
			ma = isl_multi_aff_set_tuple_id(ma, isl_dim_out,
							    isl_id_copy(id));
			map = isl_map_from_multi_aff(ma);
			cluster_map = isl_union_map_add_map(cluster_map, map);
		}
		isl_id_free(id);
	}

	return cluster_map;
}

/* Add "umap" to the schedule constraints "sc" of all types of "edge"
 * that are not isl_edge_condition or isl_edge_conditional_validity.
 */
static __isl_give isl_schedule_constraints *add_non_conditional_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *umap,
	__isl_take isl_schedule_constraints *sc)
{
	enum isl_edge_type t;

	if (!sc)
		return NULL;

	for (t = isl_edge_first; t <= isl_edge_last; ++t) {
		if (t == isl_edge_condition ||
		    t == isl_edge_conditional_validity)
			continue;
		if (!is_type(edge, t))
			continue;
		sc = isl_schedule_constraints_add(sc, t,
						    isl_union_map_copy(umap));
	}

	return sc;
}

/* Add schedule constraints of types isl_edge_condition and
 * isl_edge_conditional_validity to "sc" by applying "umap" to
 * the domains of the wrapped relations in domain and range
 * of the corresponding tagged constraints of "edge".
 */
static __isl_give isl_schedule_constraints *add_conditional_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *umap,
	__isl_take isl_schedule_constraints *sc)
{
	enum isl_edge_type t;
	isl_union_map *tagged;

	for (t = isl_edge_condition; t <= isl_edge_conditional_validity; ++t) {
		if (!is_type(edge, t))
			continue;
		if (t == isl_edge_condition)
			tagged = isl_union_map_copy(edge->tagged_condition);
		else
			tagged = isl_union_map_copy(edge->tagged_validity);
		tagged = isl_union_map_zip(tagged);
		tagged = isl_union_map_apply_domain(tagged,
					isl_union_map_copy(umap));
		tagged = isl_union_map_zip(tagged);
		sc = isl_schedule_constraints_add(sc, t, tagged);
		if (!sc)
			return NULL;
	}

	return sc;
}

/* Given a mapping "cluster_map" from the original instances to
 * the cluster instances, add schedule constraints on the clusters
 * to "sc" corresponding to the original constraints represented by "edge".
 *
 * For non-tagged dependence constraints, the cluster constraints
 * are obtained by applying "cluster_map" to the edge->map.
 *
 * For tagged dependence constraints, "cluster_map" needs to be applied
 * to the domains of the wrapped relations in domain and range
 * of the tagged dependence constraints.  Pick out the mappings
 * from these domains from "cluster_map" and construct their product.
 * This mapping can then be applied to the pair of domains.
 */
static __isl_give isl_schedule_constraints *collect_edge_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *cluster_map,
	__isl_take isl_schedule_constraints *sc)
{
	isl_union_map *umap;
	isl_space *space;
	isl_union_set *uset;
	isl_union_map *umap1, *umap2;

	if (!sc)
		return NULL;

	umap = isl_union_map_from_map(isl_map_copy(edge->map));
	umap = isl_union_map_apply_domain(umap,
				isl_union_map_copy(cluster_map));
	umap = isl_union_map_apply_range(umap,
				isl_union_map_copy(cluster_map));
	sc = add_non_conditional_constraints(edge, umap, sc);
	isl_union_map_free(umap);

	if (!sc || (!is_condition(edge) && !is_conditional_validity(edge)))
		return sc;

	space = isl_space_domain(isl_map_get_space(edge->map));
	uset = isl_union_set_from_set(isl_set_universe(space));
	umap1 = isl_union_map_copy(cluster_map);
	umap1 = isl_union_map_intersect_domain(umap1, uset);
	space = isl_space_range(isl_map_get_space(edge->map));
	uset = isl_union_set_from_set(isl_set_universe(space));
	umap2 = isl_union_map_copy(cluster_map);
	umap2 = isl_union_map_intersect_domain(umap2, uset);
	umap = isl_union_map_product(umap1, umap2);

	sc = add_conditional_constraints(edge, umap, sc);

	isl_union_map_free(umap);
	return sc;
}

/* Given a mapping "cluster_map" from the original instances to
 * the cluster instances, add schedule constraints on the clusters
 * to "sc" corresponding to all edges in "graph" between nodes that
 * belong to SCCs that are marked for merging in "scc_in_merge".
 */
static __isl_give isl_schedule_constraints *collect_constraints(
	struct isl_sched_graph *graph, int *scc_in_merge,
	__isl_keep isl_union_map *cluster_map,
	__isl_take isl_schedule_constraints *sc)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];

		if (!scc_in_merge[edge->src->scc])
			continue;
		if (!scc_in_merge[edge->dst->scc])
			continue;
		sc = collect_edge_constraints(edge, cluster_map, sc);
	}

	return sc;
}

/* Construct a dependence graph for scheduling clusters with respect
 * to each other and store the result in "merge_graph".
 * In particular, the nodes of the graph correspond to the schedule
 * dimensions of the current bands of those clusters that have been
 * marked for merging in "c".
 *
 * First construct an isl_schedule_constraints object for this domain
 * by transforming the edges in "graph" to the domain.
 * Then initialize a dependence graph for scheduling from these
 * constraints.
 */
static isl_stat init_merge_graph(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c, struct isl_sched_graph *merge_graph)
{
	isl_union_set *domain;
	isl_union_map *cluster_map;
	isl_schedule_constraints *sc;
	isl_stat r;

	domain = collect_domain(ctx, graph, c);
	sc = isl_schedule_constraints_on_domain(domain);
	if (!sc)
		return isl_stat_error;
	cluster_map = collect_cluster_map(ctx, graph, c);
	sc = collect_constraints(graph, c->scc_in_merge, cluster_map, sc);
	isl_union_map_free(cluster_map);

	r = graph_init(merge_graph, sc);

	isl_schedule_constraints_free(sc);

	return r;
}

/* Compute the maximal number of remaining schedule rows that still need
 * to be computed for the nodes that belong to clusters with the maximal
 * dimension for the current band (i.e., the band that is to be merged).
 * Only clusters that are about to be merged are considered.
 * "maxvar" is the maximal dimension for the current band.
 * "c" contains information about the clusters.
 *
 * Return the maximal number of remaining schedule rows or -1 on error.
 */
static int compute_maxvar_max_slack(int maxvar, struct isl_clustering *c)
{
	int i, j;
	int max_slack;

	max_slack = 0;
	for (i = 0; i < c->n; ++i) {
		int nvar;
		struct isl_sched_graph *scc;

		if (!c->scc_in_merge[i])
			continue;
		scc = &c->scc[i];
		nvar = scc->n_total_row - scc->band_start;
		if (nvar != maxvar)
			continue;
		for (j = 0; j < scc->n; ++j) {
			struct isl_sched_node *node = &scc->node[j];
			int slack;

			if (node_update_cmap(node) < 0)
				return -1;
			slack = node->nvar - node->rank;
			if (slack > max_slack)
				max_slack = slack;
		}
	}

	return max_slack;
}

/* If there are any clusters where the dimension of the current band
 * (i.e., the band that is to be merged) is smaller than "maxvar" and
 * if there are any nodes in such a cluster where the number
 * of remaining schedule rows that still need to be computed
 * is greater than "max_slack", then return the smallest current band
 * dimension of all these clusters.  Otherwise return the original value
 * of "maxvar".  Return -1 in case of any error.
 * Only clusters that are about to be merged are considered.
 * "c" contains information about the clusters.
 */
static int limit_maxvar_to_slack(int maxvar, int max_slack,
	struct isl_clustering *c)
{
	int i, j;

	for (i = 0; i < c->n; ++i) {
		int nvar;
		struct isl_sched_graph *scc;

		if (!c->scc_in_merge[i])
			continue;
		scc = &c->scc[i];
		nvar = scc->n_total_row - scc->band_start;
		if (nvar >= maxvar)
			continue;
		for (j = 0; j < scc->n; ++j) {
			struct isl_sched_node *node = &scc->node[j];
			int slack;

			if (node_update_cmap(node) < 0)
				return -1;
			slack = node->nvar - node->rank;
			if (slack > max_slack) {
				maxvar = nvar;
				break;
			}
		}
	}

	return maxvar;
}

/* Adjust merge_graph->maxvar based on the number of remaining schedule rows
 * that still need to be computed.  In particular, if there is a node
 * in a cluster where the dimension of the current band is smaller
 * than merge_graph->maxvar, but the number of remaining schedule rows
 * is greater than that of any node in a cluster with the maximal
 * dimension for the current band (i.e., merge_graph->maxvar),
 * then adjust merge_graph->maxvar to the (smallest) current band dimension
 * of those clusters.  Without this adjustment, the total number of
 * schedule dimensions would be increased, resulting in a skewed view
 * of the number of coincident dimensions.
 * "c" contains information about the clusters.
 *
 * If the maximize_band_depth option is set and merge_graph->maxvar is reduced,
 * then there is no point in attempting any merge since it will be rejected
 * anyway.  Set merge_graph->maxvar to zero in such cases.
 */
static isl_stat adjust_maxvar_to_slack(isl_ctx *ctx,
	struct isl_sched_graph *merge_graph, struct isl_clustering *c)
{
	int max_slack, maxvar;

	max_slack = compute_maxvar_max_slack(merge_graph->maxvar, c);
	if (max_slack < 0)
		return isl_stat_error;
	maxvar = limit_maxvar_to_slack(merge_graph->maxvar, max_slack, c);
	if (maxvar < 0)
		return isl_stat_error;

	if (maxvar < merge_graph->maxvar) {
		if (isl_options_get_schedule_maximize_band_depth(ctx))
			merge_graph->maxvar = 0;
		else
			merge_graph->maxvar = maxvar;
	}

	return isl_stat_ok;
}

/* Return the number of coincident dimensions in the current band of "graph",
 * where the nodes of "graph" are assumed to be scheduled by a single band.
 */
static int get_n_coincident(struct isl_sched_graph *graph)
{
	int i;

	for (i = graph->band_start; i < graph->n_total_row; ++i)
		if (!graph->node[0].coincident[i])
			break;

	return i - graph->band_start;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph", given that
 * coincidence should be maximized?
 *
 * If the number of coincident schedule dimensions in the merged band
 * would be less than the maximal number of coincident schedule dimensions
 * in any of the merged clusters, then the clusters should not be merged.
 */
static isl_bool ok_to_merge_coincident(struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;
	int n_coincident;
	int max_coincident;

	max_coincident = 0;
	for (i = 0; i < c->n; ++i) {
		if (!c->scc_in_merge[i])
			continue;
		n_coincident = get_n_coincident(&c->scc[i]);
		if (n_coincident > max_coincident)
			max_coincident = n_coincident;
	}

	n_coincident = get_n_coincident(merge_graph);

	return n_coincident >= max_coincident;
}

/* Return the transformation on "node" expressed by the current (and only)
 * band of "merge_graph" applied to the clusters in "c".
 *
 * First find the representation of "node" in its SCC in "c" and
 * extract the transformation expressed by the current band.
 * Then extract the transformation applied by "merge_graph"
 * to the cluster to which this SCC belongs.
 * Combine the two to obtain the complete transformation on the node.
 *
 * Note that the range of the first transformation is an anonymous space,
 * while the domain of the second is named "cluster_X".  The range
 * of the former therefore needs to be adjusted before the two
 * can be combined.
 */
static __isl_give isl_map *extract_node_transformation(isl_ctx *ctx,
	struct isl_sched_node *node, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	struct isl_sched_node *scc_node, *cluster_node;
	int start, n;
	isl_id *id;
	isl_space *space;
	isl_multi_aff *ma, *ma2;

	scc_node = graph_find_node(ctx, &c->scc[node->scc], node->space);
	start = c->scc[node->scc].band_start;
	n = c->scc[node->scc].n_total_row - start;
	ma = node_extract_partial_schedule_multi_aff(scc_node, start, n);
	space = cluster_space(&c->scc[node->scc], c->scc_cluster[node->scc]);
	cluster_node = graph_find_node(ctx, merge_graph, space);
	if (space && !cluster_node)
		isl_die(ctx, isl_error_internal, "unable to find cluster",
			space = isl_space_free(space));
	id = isl_space_get_tuple_id(space, isl_dim_set);
	ma = isl_multi_aff_set_tuple_id(ma, isl_dim_out, id);
	isl_space_free(space);
	n = merge_graph->n_total_row;
	ma2 = node_extract_partial_schedule_multi_aff(cluster_node, 0, n);
	ma = isl_multi_aff_pullback_multi_aff(ma2, ma);

	return isl_map_from_multi_aff(ma);
}

/* Give a set of distances "set", are they bounded by a small constant
 * in direction "pos"?
 * In practice, check if they are bounded by 2 by checking that there
 * are no elements with a value greater than or equal to 3 or
 * smaller than or equal to -3.
 */
static isl_bool distance_is_bounded(__isl_keep isl_set *set, int pos)
{
	isl_bool bounded;
	isl_set *test;

	if (!set)
		return isl_bool_error;

	test = isl_set_copy(set);
	test = isl_set_lower_bound_si(test, isl_dim_set, pos, 3);
	bounded = isl_set_is_empty(test);
	isl_set_free(test);

	if (bounded < 0 || !bounded)
		return bounded;

	test = isl_set_copy(set);
	test = isl_set_upper_bound_si(test, isl_dim_set, pos, -3);
	bounded = isl_set_is_empty(test);
	isl_set_free(test);

	return bounded;
}

/* Does the set "set" have a fixed (but possible parametric) value
 * at dimension "pos"?
 */
static isl_bool has_single_value(__isl_keep isl_set *set, int pos)
{
	int n;
	isl_bool single;

	if (!set)
		return isl_bool_error;
	set = isl_set_copy(set);
	n = isl_set_dim(set, isl_dim_set);
	set = isl_set_project_out(set, isl_dim_set, pos + 1, n - (pos + 1));
	set = isl_set_project_out(set, isl_dim_set, 0, pos);
	single = isl_set_is_singleton(set);
	isl_set_free(set);

	return single;
}

/* Does "map" have a fixed (but possible parametric) value
 * at dimension "pos" of either its domain or its range?
 */
static isl_bool has_singular_src_or_dst(__isl_keep isl_map *map, int pos)
{
	isl_set *set;
	isl_bool single;

	set = isl_map_domain(isl_map_copy(map));
	single = has_single_value(set, pos);
	isl_set_free(set);

	if (single < 0 || single)
		return single;

	set = isl_map_range(isl_map_copy(map));
	single = has_single_value(set, pos);
	isl_set_free(set);

	return single;
}

/* Does the edge "edge" from "graph" have bounded dependence distances
 * in the merged graph "merge_graph" of a selection of clusters in "c"?
 *
 * Extract the complete transformations of the source and destination
 * nodes of the edge, apply them to the edge constraints and
 * compute the differences.  Finally, check if these differences are bounded
 * in each direction.
 *
 * If the dimension of the band is greater than the number of
 * dimensions that can be expected to be optimized by the edge
 * (based on its weight), then also allow the differences to be unbounded
 * in the remaining dimensions, but only if either the source or
 * the destination has a fixed value in that direction.
 * This allows a statement that produces values that are used by
 * several instances of another statement to be merged with that
 * other statement.
 * However, merging such clusters will introduce an inherently
 * large proximity distance inside the merged cluster, meaning
 * that proximity distances will no longer be optimized in
 * subsequent merges.  These merges are therefore only allowed
 * after all other possible merges have been tried.
 * The first time such a merge is encountered, the weight of the edge
 * is replaced by a negative weight.  The second time (i.e., after
 * all merges over edges with a non-negative weight have been tried),
 * the merge is allowed.
 */
static isl_bool has_bounded_distances(isl_ctx *ctx, struct isl_sched_edge *edge,
	struct isl_sched_graph *graph, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i, n, n_slack;
	isl_bool bounded;
	isl_map *map, *t;
	isl_set *dist;

	map = isl_map_copy(edge->map);
	t = extract_node_transformation(ctx, edge->src, c, merge_graph);
	map = isl_map_apply_domain(map, t);
	t = extract_node_transformation(ctx, edge->dst, c, merge_graph);
	map = isl_map_apply_range(map, t);
	dist = isl_map_deltas(isl_map_copy(map));

	bounded = isl_bool_true;
	n = isl_set_dim(dist, isl_dim_set);
	n_slack = n - edge->weight;
	if (edge->weight < 0)
		n_slack -= graph->max_weight + 1;
	for (i = 0; i < n; ++i) {
		isl_bool bounded_i, singular_i;

		bounded_i = distance_is_bounded(dist, i);
		if (bounded_i < 0)
			goto error;
		if (bounded_i)
			continue;
		if (edge->weight >= 0)
			bounded = isl_bool_false;
		n_slack--;
		if (n_slack < 0)
			break;
		singular_i = has_singular_src_or_dst(map, i);
		if (singular_i < 0)
			goto error;
		if (singular_i)
			continue;
		bounded = isl_bool_false;
		break;
	}
	if (!bounded && i >= n && edge->weight >= 0)
		edge->weight -= graph->max_weight + 1;
	isl_map_free(map);
	isl_set_free(dist);

	return bounded;
error:
	isl_map_free(map);
	isl_set_free(dist);
	return isl_bool_error;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph"?
 * "graph" is the original dependence graph, while "c" records
 * which SCCs are involved in the latest merge.
 *
 * In particular, is there at least one proximity constraint
 * that is optimized by the merge?
 *
 * A proximity constraint is considered to be optimized
 * if the dependence distances are small.
 */
static isl_bool ok_to_merge_proximity(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		isl_bool bounded;

		if (!is_proximity(edge))
			continue;
		if (!c->scc_in_merge[edge->src->scc])
			continue;
		if (!c->scc_in_merge[edge->dst->scc])
			continue;
		if (c->scc_cluster[edge->dst->scc] ==
		    c->scc_cluster[edge->src->scc])
			continue;
		bounded = has_bounded_distances(ctx, edge, graph, c,
						merge_graph);
		if (bounded < 0 || bounded)
			return bounded;
	}

	return isl_bool_false;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph"?
 * "graph" is the original dependence graph, while "c" records
 * which SCCs are involved in the latest merge.
 *
 * If the current band is empty, then the clusters should not be merged.
 *
 * If the band depth should be maximized and the merge schedule
 * is incomplete (meaning that the dimension of some of the schedule
 * bands in the original schedule will be reduced), then the clusters
 * should not be merged.
 *
 * If the schedule_maximize_coincidence option is set, then check that
 * the number of coincident schedule dimensions is not reduced.
 *
 * Finally, only allow the merge if at least one proximity
 * constraint is optimized.
 */
static isl_bool ok_to_merge(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c, struct isl_sched_graph *merge_graph)
{
	if (merge_graph->n_total_row == merge_graph->band_start)
		return isl_bool_false;

	if (isl_options_get_schedule_maximize_band_depth(ctx) &&
	    merge_graph->n_total_row < merge_graph->maxvar)
		return isl_bool_false;

	if (isl_options_get_schedule_maximize_coincidence(ctx)) {
		isl_bool ok;

		ok = ok_to_merge_coincident(c, merge_graph);
		if (ok < 0 || !ok)
			return ok;
	}

	return ok_to_merge_proximity(ctx, graph, c, merge_graph);
}

/* Apply the schedule in "t_node" to the "n" rows starting at "first"
 * of the schedule in "node" and return the result.
 *
 * That is, essentially compute
 *
 *	T * N(first:first+n-1)
 *
 * taking into account the constant term and the parameter coefficients
 * in "t_node".
 */
static __isl_give isl_mat *node_transformation(isl_ctx *ctx,
	struct isl_sched_node *t_node, struct isl_sched_node *node,
	int first, int n)
{
	int i, j;
	isl_mat *t;
	int n_row, n_col, n_param, n_var;

	n_param = node->nparam;
	n_var = node->nvar;
	n_row = isl_mat_rows(t_node->sched);
	n_col = isl_mat_cols(node->sched);
	t = isl_mat_alloc(ctx, n_row, n_col);
	if (!t)
		return NULL;
	for (i = 0; i < n_row; ++i) {
		isl_seq_cpy(t->row[i], t_node->sched->row[i], 1 + n_param);
		isl_seq_clr(t->row[i] + 1 + n_param, n_var);
		for (j = 0; j < n; ++j)
			isl_seq_addmul(t->row[i],
					t_node->sched->row[i][1 + n_param + j],
					node->sched->row[first + j],
					1 + n_param + n_var);
	}
	return t;
}

/* Apply the cluster schedule in "t_node" to the current band
 * schedule of the nodes in "graph".
 *
 * In particular, replace the rows starting at band_start
 * by the result of applying the cluster schedule in "t_node"
 * to the original rows.
 *
 * The coincidence of the schedule is determined by the coincidence
 * of the cluster schedule.
 */
static isl_stat transform(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_sched_node *t_node)
{
	int i, j;
	int n_new;
	int start, n;

	start = graph->band_start;
	n = graph->n_total_row - start;

	n_new = isl_mat_rows(t_node->sched);
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		isl_mat *t;

		t = node_transformation(ctx, t_node, node, start, n);
		node->sched = isl_mat_drop_rows(node->sched, start, n);
		node->sched = isl_mat_concat(node->sched, t);
		node->sched_map = isl_map_free(node->sched_map);
		if (!node->sched)
			return isl_stat_error;
		for (j = 0; j < n_new; ++j)
			node->coincident[start + j] = t_node->coincident[j];
	}
	graph->n_total_row -= n;
	graph->n_row -= n;
	graph->n_total_row += n_new;
	graph->n_row += n_new;

	return isl_stat_ok;
}

/* Merge the clusters marked for merging in "c" into a single
 * cluster using the cluster schedule in the current band of "merge_graph".
 * The representative SCC for the new cluster is the SCC with
 * the smallest index.
 *
 * The current band schedule of each SCC in the new cluster is obtained
 * by applying the schedule of the corresponding original cluster
 * to the original band schedule.
 * All SCCs in the new cluster have the same number of schedule rows.
 */
static isl_stat merge(isl_ctx *ctx, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;
	int cluster = -1;
	isl_space *space;

	for (i = 0; i < c->n; ++i) {
		struct isl_sched_node *node;

		if (!c->scc_in_merge[i])
			continue;
		if (cluster < 0)
			cluster = i;
		space = cluster_space(&c->scc[i], c->scc_cluster[i]);
		if (!space)
			return isl_stat_error;
		node = graph_find_node(ctx, merge_graph, space);
		isl_space_free(space);
		if (!node)
			isl_die(ctx, isl_error_internal,
				"unable to find cluster",
				return isl_stat_error);
		if (transform(ctx, &c->scc[i], node) < 0)
			return isl_stat_error;
		c->scc_cluster[i] = cluster;
	}

	return isl_stat_ok;
}

/* Try and merge the clusters of SCCs marked in c->scc_in_merge
 * by scheduling the current cluster bands with respect to each other.
 *
 * Construct a dependence graph with a space for each cluster and
 * with the coordinates of each space corresponding to the schedule
 * dimensions of the current band of that cluster.
 * Construct a cluster schedule in this cluster dependence graph and
 * apply it to the current cluster bands if it is applicable
 * according to ok_to_merge.
 *
 * If the number of remaining schedule dimensions in a cluster
 * with a non-maximal current schedule dimension is greater than
 * the number of remaining schedule dimensions in clusters
 * with a maximal current schedule dimension, then restrict
 * the number of rows to be computed in the cluster schedule
 * to the minimal such non-maximal current schedule dimension.
 * Do this by adjusting merge_graph.maxvar.
 *
 * Return isl_bool_true if the clusters have effectively been merged
 * into a single cluster.
 *
 * Note that since the standard scheduling algorithm minimizes the maximal
 * distance over proximity constraints, the proximity constraints between
 * the merged clusters may not be optimized any further than what is
 * sufficient to bring the distances within the limits of the internal
 * proximity constraints inside the individual clusters.
 * It may therefore make sense to perform an additional translation step
 * to bring the clusters closer to each other, while maintaining
 * the linear part of the merging schedule found using the standard
 * scheduling algorithm.
 */
static isl_bool try_merge(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	struct isl_sched_graph merge_graph = { 0 };
	isl_bool merged;

	if (init_merge_graph(ctx, graph, c, &merge_graph) < 0)
		goto error;

	if (compute_maxvar(&merge_graph) < 0)
		goto error;
	if (adjust_maxvar_to_slack(ctx, &merge_graph,c) < 0)
		goto error;
	if (compute_schedule_wcc_band(ctx, &merge_graph) < 0)
		goto error;
	merged = ok_to_merge(ctx, graph, c, &merge_graph);
	if (merged && merge(ctx, c, &merge_graph) < 0)
		goto error;

	graph_free(ctx, &merge_graph);
	return merged;
error:
	graph_free(ctx, &merge_graph);
	return isl_bool_error;
}

/* Is there any edge marked "no_merge" between two SCCs that are
 * about to be merged (i.e., that are set in "scc_in_merge")?
 * "merge_edge" is the proximity edge along which the clusters of SCCs
 * are going to be merged.
 *
 * If there is any edge between two SCCs with a negative weight,
 * while the weight of "merge_edge" is non-negative, then this
 * means that the edge was postponed.  "merge_edge" should then
 * also be postponed since merging along the edge with negative weight should
 * be postponed until all edges with non-negative weight have been tried.
 * Replace the weight of "merge_edge" by a negative weight as well and
 * tell the caller not to attempt a merge.
 */
static int any_no_merge(struct isl_sched_graph *graph, int *scc_in_merge,
	struct isl_sched_edge *merge_edge)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];

		if (!scc_in_merge[edge->src->scc])
			continue;
		if (!scc_in_merge[edge->dst->scc])
			continue;
		if (edge->no_merge)
			return 1;
		if (merge_edge->weight >= 0 && edge->weight < 0) {
			merge_edge->weight -= graph->max_weight + 1;
			return 1;
		}
	}

	return 0;
}

/* Merge the two clusters in "c" connected by the edge in "graph"
 * with index "edge" into a single cluster.
 * If it turns out to be impossible to merge these two clusters,
 * then mark the edge as "no_merge" such that it will not be
 * considered again.
 *
 * First mark all SCCs that need to be merged.  This includes the SCCs
 * in the two clusters, but it may also include the SCCs
 * of intermediate clusters.
 * If there is already a no_merge edge between any pair of such SCCs,
 * then simply mark the current edge as no_merge as well.
 * Likewise, if any of those edges was postponed by has_bounded_distances,
 * then postpone the current edge as well.
 * Otherwise, try and merge the clusters and mark "edge" as "no_merge"
 * if the clusters did not end up getting merged, unless the non-merge
 * is due to the fact that the edge was postponed.  This postponement
 * can be recognized by a change in weight (from non-negative to negative).
 */
static isl_stat merge_clusters_along_edge(isl_ctx *ctx,
	struct isl_sched_graph *graph, int edge, struct isl_clustering *c)
{
	isl_bool merged;
	int edge_weight = graph->edge[edge].weight;

	if (mark_merge_sccs(ctx, graph, edge, c) < 0)
		return isl_stat_error;

	if (any_no_merge(graph, c->scc_in_merge, &graph->edge[edge]))
		merged = isl_bool_false;
	else
		merged = try_merge(ctx, graph, c);
	if (merged < 0)
		return isl_stat_error;
	if (!merged && edge_weight == graph->edge[edge].weight)
		graph->edge[edge].no_merge = 1;

	return isl_stat_ok;
}

/* Does "node" belong to the cluster identified by "cluster"?
 */
static int node_cluster_exactly(struct isl_sched_node *node, int cluster)
{
	return node->cluster == cluster;
}

/* Does "edge" connect two nodes belonging to the cluster
 * identified by "cluster"?
 */
static int edge_cluster_exactly(struct isl_sched_edge *edge, int cluster)
{
	return edge->src->cluster == cluster && edge->dst->cluster == cluster;
}

/* Swap the schedule of "node1" and "node2".
 * Both nodes have been derived from the same node in a common parent graph.
 * Since the "coincident" field is shared with that node
 * in the parent graph, there is no need to also swap this field.
 */
static void swap_sched(struct isl_sched_node *node1,
	struct isl_sched_node *node2)
{
	isl_mat *sched;
	isl_map *sched_map;

	sched = node1->sched;
	node1->sched = node2->sched;
	node2->sched = sched;

	sched_map = node1->sched_map;
	node1->sched_map = node2->sched_map;
	node2->sched_map = sched_map;
}

/* Copy the current band schedule from the SCCs that form the cluster
 * with index "pos" to the actual cluster at position "pos".
 * By construction, the index of the first SCC that belongs to the cluster
 * is also "pos".
 *
 * The order of the nodes inside both the SCCs and the cluster
 * is assumed to be same as the order in the original "graph".
 *
 * Since the SCC graphs will no longer be used after this function,
 * the schedules are actually swapped rather than copied.
 */
static isl_stat copy_partial(struct isl_sched_graph *graph,
	struct isl_clustering *c, int pos)
{
	int i, j;

	c->cluster[pos].n_total_row = c->scc[pos].n_total_row;
	c->cluster[pos].n_row = c->scc[pos].n_row;
	c->cluster[pos].maxvar = c->scc[pos].maxvar;
	j = 0;
	for (i = 0; i < graph->n; ++i) {
		int k;
		int s;

		if (graph->node[i].cluster != pos)
			continue;
		s = graph->node[i].scc;
		k = c->scc_node[s]++;
		swap_sched(&c->cluster[pos].node[j], &c->scc[s].node[k]);
		if (c->scc[s].maxvar > c->cluster[pos].maxvar)
			c->cluster[pos].maxvar = c->scc[s].maxvar;
		++j;
	}

	return isl_stat_ok;
}

/* Is there a (conditional) validity dependence from node[j] to node[i],
 * forcing node[i] to follow node[j] or do the nodes belong to the same
 * cluster?
 */
static isl_bool node_follows_strong_or_same_cluster(int i, int j, void *user)
{
	struct isl_sched_graph *graph = user;

	if (graph->node[i].cluster == graph->node[j].cluster)
		return isl_bool_true;
	return graph_has_validity_edge(graph, &graph->node[j], &graph->node[i]);
}

/* Extract the merged clusters of SCCs in "graph", sort them, and
 * store them in c->clusters.  Update c->scc_cluster accordingly.
 *
 * First keep track of the cluster containing the SCC to which a node
 * belongs in the node itself.
 * Then extract the clusters into c->clusters, copying the current
 * band schedule from the SCCs that belong to the cluster.
 * Do this only once per cluster.
 *
 * Finally, topologically sort the clusters and update c->scc_cluster
 * to match the new scc numbering.  While the SCCs were originally
 * sorted already, some SCCs that depend on some other SCCs may
 * have been merged with SCCs that appear before these other SCCs.
 * A reordering may therefore be required.
 */
static isl_stat extract_clusters(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;

	for (i = 0; i < graph->n; ++i)
		graph->node[i].cluster = c->scc_cluster[graph->node[i].scc];

	for (i = 0; i < graph->scc; ++i) {
		if (c->scc_cluster[i] != i)
			continue;
		if (extract_sub_graph(ctx, graph, &node_cluster_exactly,
				&edge_cluster_exactly, i, &c->cluster[i]) < 0)
			return isl_stat_error;
		c->cluster[i].src_scc = -1;
		c->cluster[i].dst_scc = -1;
		if (copy_partial(graph, c, i) < 0)
			return isl_stat_error;
	}

	if (detect_ccs(ctx, graph, &node_follows_strong_or_same_cluster) < 0)
		return isl_stat_error;
	for (i = 0; i < graph->n; ++i)
		c->scc_cluster[graph->node[i].scc] = graph->node[i].cluster;

	return isl_stat_ok;
}

/* Compute weights on the proximity edges of "graph" that can
 * be used by find_proximity to find the most appropriate
 * proximity edge to use to merge two clusters in "c".
 * The weights are also used by has_bounded_distances to determine
 * whether the merge should be allowed.
 * Store the maximum of the computed weights in graph->max_weight.
 *
 * The computed weight is a measure for the number of remaining schedule
 * dimensions that can still be completely aligned.
 * In particular, compute the number of equalities between
 * input dimensions and output dimensions in the proximity constraints.
 * The directions that are already handled by outer schedule bands
 * are projected out prior to determining this number.
 *
 * Edges that will never be considered by find_proximity are ignored.
 */
static isl_stat compute_weights(struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;

	graph->max_weight = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		struct isl_sched_node *src = edge->src;
		struct isl_sched_node *dst = edge->dst;
		isl_basic_map *hull;
		int n_in, n_out;

		if (!is_proximity(edge))
			continue;
		if (bad_cluster(&c->scc[edge->src->scc]) ||
		    bad_cluster(&c->scc[edge->dst->scc]))
			continue;
		if (c->scc_cluster[edge->dst->scc] ==
		    c->scc_cluster[edge->src->scc])
			continue;

		hull = isl_map_affine_hull(isl_map_copy(edge->map));
		hull = isl_basic_map_transform_dims(hull, isl_dim_in, 0,
						    isl_mat_copy(src->ctrans));
		hull = isl_basic_map_transform_dims(hull, isl_dim_out, 0,
						    isl_mat_copy(dst->ctrans));
		hull = isl_basic_map_project_out(hull,
						isl_dim_in, 0, src->rank);
		hull = isl_basic_map_project_out(hull,
						isl_dim_out, 0, dst->rank);
		hull = isl_basic_map_remove_divs(hull);
		n_in = isl_basic_map_dim(hull, isl_dim_in);
		n_out = isl_basic_map_dim(hull, isl_dim_out);
		hull = isl_basic_map_drop_constraints_not_involving_dims(hull,
							isl_dim_in, 0, n_in);
		hull = isl_basic_map_drop_constraints_not_involving_dims(hull,
							isl_dim_out, 0, n_out);
		if (!hull)
			return isl_stat_error;
		edge->weight = hull->n_eq;
		isl_basic_map_free(hull);

		if (edge->weight > graph->max_weight)
			graph->max_weight = edge->weight;
	}

	return isl_stat_ok;
}

/* Call compute_schedule_finish_band on each of the clusters in "c"
 * in their topological order.  This order is determined by the scc
 * fields of the nodes in "graph".
 * Combine the results in a sequence expressing the topological order.
 *
 * If there is only one cluster left, then there is no need to introduce
 * a sequence node.  Also, in this case, the cluster necessarily contains
 * the SCC at position 0 in the original graph and is therefore also
 * stored in the first cluster of "c".
 */
static __isl_give isl_schedule_node *finish_bands_clustering(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (graph->scc == 1)
		return compute_schedule_finish_band(node, &c->cluster[0], 0);

	ctx = isl_schedule_node_get_ctx(node);

	filters = extract_sccs(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);

	for (i = 0; i < graph->scc; ++i) {
		int j = c->scc_cluster[i];
		node = isl_schedule_node_child(node, i);
		node = isl_schedule_node_child(node, 0);
		node = compute_schedule_finish_band(node, &c->cluster[j], 0);
		node = isl_schedule_node_parent(node);
		node = isl_schedule_node_parent(node);
	}

	return node;
}

/* Compute a schedule for a connected dependence graph by first considering
 * each strongly connected component (SCC) in the graph separately and then
 * incrementally combining them into clusters.
 * Return the updated schedule node.
 *
 * Initially, each cluster consists of a single SCC, each with its
 * own band schedule.  The algorithm then tries to merge pairs
 * of clusters along a proximity edge until no more suitable
 * proximity edges can be found.  During this merging, the schedule
 * is maintained in the individual SCCs.
 * After the merging is completed, the full resulting clusters
 * are extracted and in finish_bands_clustering,
 * compute_schedule_finish_band is called on each of them to integrate
 * the band into "node" and to continue the computation.
 *
 * compute_weights initializes the weights that are used by find_proximity.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc_clustering(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;
	struct isl_clustering c;
	int i;

	ctx = isl_schedule_node_get_ctx(node);

	if (clustering_init(ctx, &c, graph) < 0)
		goto error;

	if (compute_weights(graph, &c) < 0)
		goto error;

	for (;;) {
		i = find_proximity(graph, &c);
		if (i < 0)
			goto error;
		if (i >= graph->n_edge)
			break;
		if (merge_clusters_along_edge(ctx, graph, i, &c) < 0)
			goto error;
	}

	if (extract_clusters(ctx, graph, &c) < 0)
		goto error;

	node = finish_bands_clustering(node, graph, &c);

	clustering_free(ctx, &c);
	return node;
error:
	clustering_free(ctx, &c);
	return isl_schedule_node_free(node);
}

/* Compute a schedule for a connected dependence graph and return
 * the updated schedule node.
 *
 * If Feautrier's algorithm is selected, we first recursively try to satisfy
 * as many validity dependences as possible. When all validity dependences
 * are satisfied we extend the schedule to a full-dimensional schedule.
 *
 * Call compute_schedule_wcc_whole or compute_schedule_wcc_clustering
 * depending on whether the user has selected the option to try and
 * compute a schedule for the entire (weakly connected) component first.
 * If there is only a single strongly connected component (SCC), then
 * there is no point in trying to combine SCCs
 * in compute_schedule_wcc_clustering, so compute_schedule_wcc_whole
 * is called instead.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (detect_sccs(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	if (compute_maxvar(graph) < 0)
		return isl_schedule_node_free(node);

	if (need_feautrier_step(ctx, graph))
		return compute_schedule_wcc_feautrier(node, graph);

	if (graph->scc <= 1 || isl_options_get_schedule_whole_component(ctx))
		return compute_schedule_wcc_whole(node, graph);
	else
		return compute_schedule_wcc_clustering(node, graph);
}

/* Compute a schedule for each group of nodes identified by node->scc
 * separately and then combine them in a sequence node (or as set node
 * if graph->weak is set) inserted at position "node" of the schedule tree.
 * Return the updated schedule node.
 *
 * If "wcc" is set then each of the groups belongs to a single
 * weakly connected component in the dependence graph so that
 * there is no need for compute_sub_schedule to look for weakly
 * connected components.
 */
static __isl_give isl_schedule_node *compute_component_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int wcc)
{
	int component;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;
	ctx = isl_schedule_node_get_ctx(node);

	filters = extract_sccs(ctx, graph);
	if (graph->weak)
		node = isl_schedule_node_insert_set(node, filters);
	else
		node = isl_schedule_node_insert_sequence(node, filters);

	for (component = 0; component < graph->scc; ++component) {
		node = isl_schedule_node_child(node, component);
		node = isl_schedule_node_child(node, 0);
		node = compute_sub_schedule(node, ctx, graph,
				    &node_scc_exactly,
				    &edge_scc_exactly, component, wcc);
		node = isl_schedule_node_parent(node);
		node = isl_schedule_node_parent(node);
	}

	return node;
}

/* Compute a schedule for the given dependence graph and insert it at "node".
 * Return the updated schedule node.
 *
 * We first check if the graph is connected (through validity and conditional
 * validity dependences) and, if not, compute a schedule
 * for each component separately.
 * If the schedule_serialize_sccs option is set, then we check for strongly
 * connected components instead and compute a separate schedule for
 * each such strongly connected component.
 */
static __isl_give isl_schedule_node *compute_schedule(isl_schedule_node *node,
	struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (isl_options_get_schedule_serialize_sccs(ctx)) {
		if (detect_sccs(ctx, graph) < 0)
			return isl_schedule_node_free(node);
	} else {
		if (detect_wccs(ctx, graph) < 0)
			return isl_schedule_node_free(node);
	}

	if (graph->scc > 1)
		return compute_component_schedule(node, graph, 1);

	return compute_schedule_wcc(node, graph);
}

/* Compute a schedule on sc->domain that respects the given schedule
 * constraints.
 *
 * In particular, the schedule respects all the validity dependences.
 * If the default isl scheduling algorithm is used, it tries to minimize
 * the dependence distances over the proximity dependences.
 * If Feautrier's scheduling algorithm is used, the proximity dependence
 * distances are only minimized during the extension to a full-dimensional
 * schedule.
 *
 * If there are any condition and conditional validity dependences,
 * then the conditional validity dependences may be violated inside
 * a tilable band, provided they have no adjacent non-local
 * condition dependences.
 */
__isl_give isl_schedule *isl_schedule_constraints_compute_schedule(
	__isl_take isl_schedule_constraints *sc)
{
	isl_ctx *ctx = isl_schedule_constraints_get_ctx(sc);
	struct isl_sched_graph graph = { 0 };
	isl_schedule *sched;
	isl_schedule_node *node;
	isl_union_set *domain;

	sc = isl_schedule_constraints_align_params(sc);

	domain = isl_schedule_constraints_get_domain(sc);
	if (isl_union_set_n_set(domain) == 0) {
		isl_schedule_constraints_free(sc);
		return isl_schedule_from_domain(domain);
	}

	if (graph_init(&graph, sc) < 0)
		domain = isl_union_set_free(domain);

	node = isl_schedule_node_from_domain(domain);
	node = isl_schedule_node_child(node, 0);
	if (graph.n > 0)
		node = compute_schedule(node, &graph);
	sched = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	graph_free(ctx, &graph);
	isl_schedule_constraints_free(sc);

	return sched;
}

/* Compute a schedule for the given union of domains that respects
 * all the validity dependences and minimizes
 * the dependence distances over the proximity dependences.
 *
 * This function is kept for backward compatibility.
 */
__isl_give isl_schedule *isl_union_set_compute_schedule(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_map *validity,
	__isl_take isl_union_map *proximity)
{
	isl_schedule_constraints *sc;

	sc = isl_schedule_constraints_on_domain(domain);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);

	return isl_schedule_constraints_compute_schedule(sc);
}
