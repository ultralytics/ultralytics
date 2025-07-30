#include <isl/map_type.h>

/* Treat "bset" as a basic map.
 * Internally, isl_basic_set is defined to isl_basic_map, so in practice,
 * this function performs a redundant cast.
 */
static __isl_give isl_basic_map *bset_to_bmap(__isl_take isl_basic_set *bset)
{
	return (isl_basic_map *) bset;
}
