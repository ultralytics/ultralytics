domain: "[n] -> { S1[i0, i1, i2] : i0 >= 1 and i0 <= n and i1 >= 1 and i1 <= n and i2 >= 1 and i2 <= n; S2[i0, i1, i2] : i0 >= 1 and i0 <= n and i1 >= 1 and i1 <= n and i2 >= 1 and i2 <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i0, i1, i2] -> [(i0 - i1)]; S2[i0, i1, i2] -> [(1 + i0 - i1)] }, { S1[i0, i1, i2] -> [(i0 + i1)]; S2[i0, i1, i2] -> [(2 + i0 + i1)] }, { S1[i0, i1, i2] -> [(i0 + i1 + 2i2)]; S2[i0, i1, i2] -> [(i2)] }]"
    options: "[n] -> { separate[i0] }"
