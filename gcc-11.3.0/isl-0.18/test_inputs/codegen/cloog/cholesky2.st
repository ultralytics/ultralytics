domain: "[M] -> { S4[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M; S5[i0, i1, i2] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M and i2 >= 1 and i2 <= -1 + i0; S6[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 + i0 and i1 <= M; S3[i0] : i0 >= 1 and i0 <= M; S2[i0, i1] : i0 >= 1 and i0 <= M and i1 >= 1 and i1 <= -1 + i0; S1[i0] : i0 >= 1 and i0 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0] -> [(0)]; S3[i0] -> [(-2 + 3i0)]; S4[i0, i1] -> [(0)]; S5[i0, i1, i2] -> [(-1 + 3i2)]; S2[i0, i1] -> [(3i1)]; S6[i0, i1] -> [(-1 + 3i0)] }, { S1[i0] -> [(i0)]; S3[i0] -> [(0)]; S4[i0, i1] -> [(i0)]; S5[i0, i1, i2] -> [(i1)]; S2[i0, i1] -> [(i0)]; S6[i0, i1] -> [(i1)] }, { S1[i0] -> [(0)]; S3[i0] -> [(0)]; S4[i0, i1] -> [(i1)]; S5[i0, i1, i2] -> [(i2)]; S2[i0, i1] -> [(0)]; S6[i0, i1] -> [(0)] }]"
    options: "[M] -> { separate[i0] }"
