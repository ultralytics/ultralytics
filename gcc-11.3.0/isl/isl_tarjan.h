#ifndef ISL_TARJAN_H
#define ISL_TARJAN_H

/* Structure for representing the nodes in the graph being traversed
 * using Tarjan's algorithm.
 * index represents the order in which nodes are visited.
 * min_index is the index of the root of a (sub)component.
 * on_stack indicates whether the node is currently on the stack.
 */
struct isl_tarjan_node {
	int index;
	int min_index;
	int on_stack;
};

/* Structure for representing the graph being traversed
 * using Tarjan's algorithm.
 * len is the number of nodes
 * node is an array of nodes
 * stack contains the nodes on the path from the root to the current node
 * sp is the stack pointer
 * index is the index of the last node visited
 * order contains the elements of the components separated by -1
 * op represents the current position in order
 */
struct isl_tarjan_graph {
	int len;
	struct isl_tarjan_node *node;
	int *stack;
	int sp;
	int index;
	int *order;
	int op;
};

struct isl_tarjan_graph *isl_tarjan_graph_init(isl_ctx *ctx, int len,
	isl_bool (*follows)(int i, int j, void *user), void *user);
struct isl_tarjan_graph *isl_tarjan_graph_component(isl_ctx *ctx, int len,
	int node, isl_bool (*follows)(int i, int j, void *user), void *user);
struct isl_tarjan_graph *isl_tarjan_graph_free(struct isl_tarjan_graph *g);

#endif
