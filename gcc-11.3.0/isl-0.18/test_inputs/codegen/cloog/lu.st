domain: "[n] -> { S1[i0, i1] : i0 >= 1 and i0 <= n and i1 >= 1 + i0 and i1 <= n; S2[i0, i1, i2] : i0 >= 1 and i0 <= n and i1 >= 1 + i0 and i1 <= n and i2 >= 1 + i0 and i2 <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S2[i0, i1, i2] -> [(i2)]; S1[i0, i1] -> [(i0)] }, { S2[i0, i1, i2] -> [(i1)]; S1[i0, i1] -> [(n)] }]"
    options: "[n] -> { separate[i0] }"
