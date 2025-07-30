domain: "[M] -> { S1[i0, i0, M, i3] : i0 >= 1 and i0 <= M and i3 >= 1 + i0 and i3 <= M; S2[i0, i1, i2, i2, i0] : i1 >= 1 and i1 <= M and i2 >= 1 + i1 and i2 <= M and i1 <= -1 + i0 and i0 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2, i3] -> [(i0)]; S2[i0, i1, i2, i3, i4] -> [(i0)] }, { S1[i0, i1, i2, i3] -> [(i1)]; S2[i0, i1, i2, i3, i4] -> [(i1)] }, { S1[i0, i1, i2, i3] -> [(i2)]; S2[i0, i1, i2, i3, i4] -> [(i2)] }, { S1[i0, i1, i2, i3] -> [(i3)]; S2[i0, i1, i2, i3, i4] -> [(i3)] }, { S1[i0, i1, i2, i3] -> [(0)]; S2[i0, i1, i2, i3, i4] -> [(i4)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1, i2, i3] }"
      - filter: "[M] -> { S2[i0, i1, i2, i3, i4] }"
