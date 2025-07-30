/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <stdlib.h>
#include <isl/ctx.h>
#include <isl_tarjan.h>

struct isl_tarjan_graph *isl_tarjan_graph_free(struct isl_tarjan_graph *g)
{
	if (!g)
		return NULL;
	free(g->node);
	free(g->stack);
	free(g->order);
	free(g);
	return NULL;
}

static struct isl_tarjan_graph *isl_tarjan_graph_alloc(isl_ctx *ctx, int len)
{
	struct isl_tarjan_graph *g;
	int i;

	g = isl_calloc_type(ctx, struct isl_tarjan_graph);
	if (!g)
		return NULL;
	g->len = len;
	g->node = isl_alloc_array(ctx, struct isl_tarjan_node, len);
	if (len && !g->node)
		goto error;
	for (i = 0; i < len; ++i)
		g->node[i].index = -1;
	g->stack = isl_alloc_array(ctx, int, len);
	if (len && !g->stack)
		goto error;
	g->order = isl_alloc_array(ctx, int, 2 * len);
	if (len && !g->order)
		goto error;

	g->sp = 0;
	g->index = 0;
	g->op = 0;

	return g;
error:
	isl_tarjan_graph_free(g);
	return NULL;
}

/* Perform Tarjan's algorithm for computing the strongly connected components
 * in the graph with g->len nodes and with edges defined by "follows".
 */
static isl_stat isl_tarjan_components(struct isl_tarjan_graph *g, int i,
	isl_bool (*follows)(int i, int j, void *user), void *user)
{
	int j;

	g->node[i].index = g->index;
	g->node[i].min_index = g->index;
	g->node[i].on_stack = 1;
	g->index++;
	g->stack[g->sp++] = i;

	for (j = g->len - 1; j >= 0; --j) {
		isl_bool f;

		if (j == i)
			continue;
		if (g->node[j].index >= 0 &&
			(!g->node[j].on_stack ||
			 g->node[j].index > g->node[i].min_index))
			continue;

		f = follows(i, j, user);
		if (f < 0)
			return isl_stat_error;
		if (!f)
			continue;

		if (g->node[j].index < 0) {
			isl_tarjan_components(g, j, follows, user);
			if (g->node[j].min_index < g->node[i].min_index)
				g->node[i].min_index = g->node[j].min_index;
		} else if (g->node[j].index < g->node[i].min_index)
			g->node[i].min_index = g->node[j].index;
	}

	if (g->node[i].index != g->node[i].min_index)
		return isl_stat_ok;

	do {
		j = g->stack[--g->sp];
		g->node[j].on_stack = 0;
		g->order[g->op++] = j;
	} while (j != i);
	g->order[g->op++] = -1;

	return isl_stat_ok;
}

/* Decompose the graph with "len" nodes and edges defined by "follows"
 * into strongly connected components (SCCs).
 * follows(i, j, user) should return 1 if "i" follows "j" and 0 otherwise.
 * It should return -1 on error.
 *
 * If SCC a contains a node i that follows a node j in another SCC b
 * (i.e., follows(i, j, user) returns 1), then SCC a will appear after SCC b
 * in the result.
 */
struct isl_tarjan_graph *isl_tarjan_graph_init(isl_ctx *ctx, int len,
	isl_bool (*follows)(int i, int j, void *user), void *user)
{
	int i;
	struct isl_tarjan_graph *g = NULL;

	g = isl_tarjan_graph_alloc(ctx, len);
	if (!g)
		return NULL;
	for (i = len - 1; i >= 0; --i) {
		if (g->node[i].index >= 0)
			continue;
		if (isl_tarjan_components(g, i, follows, user) < 0)
			return isl_tarjan_graph_free(g);
	}

	return g;
}

/* Decompose the graph with "len" nodes and edges defined by "follows"
 * into the strongly connected component (SCC) that contains "node"
 * as well as all SCCs that are followed by this SCC.
 * follows(i, j, user) should return 1 if "i" follows "j" and 0 otherwise.
 * It should return -1 on error.
 *
 * The SCC containing "node" will appear as the last component
 * in g->order.
 */
struct isl_tarjan_graph *isl_tarjan_graph_component(isl_ctx *ctx, int len,
	int node, isl_bool (*follows)(int i, int j, void *user), void *user)
{
	struct isl_tarjan_graph *g;

	g = isl_tarjan_graph_alloc(ctx, len);
	if (!g)
		return NULL;
	if (isl_tarjan_components(g, node, follows, user) < 0)
		return isl_tarjan_graph_free(g);

	return g;
}
