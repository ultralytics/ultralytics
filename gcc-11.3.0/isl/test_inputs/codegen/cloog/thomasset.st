domain: "[n] -> { S1[i, j] : i <= n and i >= 1 and 3j <= -1 + i and 3j >= -3 + i; S2[i, j, 0, p, q] : i <= n and j <= n and j >= 1 and i >= 1 and 3q <= j and 3q >= -2 + j and 3p <= n and 3p >= -2 + n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S2[i0, i1, i2, i3, i4] -> [(i2 + i3 + i4)]; S1[i0, i1] -> [(i1)] }, { S2[i0, i1, i2, i3, i4] -> [(-1 + i0)]; S1[i0, i1] -> [(0)] }]"
    options: "[n] -> { separate[i0] }"
