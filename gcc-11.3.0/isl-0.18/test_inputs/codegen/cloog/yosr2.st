domain: "[M] -> { S4[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M; S3[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M and i2 >= 1 and i2 <= -1 + i0; S1[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= -1 + i0; S2[i0] : i0 >= 1 and i0 <= M }"
child:
  context: "[M] -> { [] : M >= 2 }"
  child:
    schedule: "[M] -> [{ S2[i0] -> [(0)]; S1[i0, i1] -> [(i0)]; S4[i0, i1] -> [(i1)]; S3[i0, i1, i2] -> [(i0)] }]"
    options: "[M] -> { separate[i0] }"
