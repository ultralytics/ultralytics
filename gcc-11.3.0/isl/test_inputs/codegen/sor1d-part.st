# Check that a proper upper bound is generated for the outer loop.
# Earlier versions of isl would generate an incorrect bound.
domain: "[M, N] -> { S2[i0, i1] : i1 >= 0 and  200i1 >= -193 - N + 100i0 and i0 >= 0 and 200i1 <= -204 + 2M + N and 2i1 <= -1 + i0 and 100i1 >= -94 - N + 50i0 and 100i1 >= -94 - N }"
child:
  context: "[M, N] -> { [] : M >= 0 and N >= 4 }"
  child:
    schedule: "[M, N] -> [{ S2[i0, i1] -> [(i0 + i1)]}]"
    options: "[M, N] -> { separate[i0] }"
    child:
          schedule: "[M, N] -> [{ S2[i0, i1] -> [(i1)]}]"
