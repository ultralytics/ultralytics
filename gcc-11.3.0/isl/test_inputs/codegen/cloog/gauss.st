domain: "[M] -> { S2[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M and i2 >= 1 + i0 and i2 <= M; S1[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= -1 + i0 and i2 >= 1 + i0 and i2 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i0)]; S2[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i2)]; S2[i0, i1, i2] -> [(i2)] }]"
    options: "[M] -> { separate[i0] }"
