domain: "[M] -> { S4[i0, 0] : i0 >= 2 and M >= 3 and i0 <= M; S3[1, 0] : M >= 3; S2[i0, 1, i2] : i2 >= 1 + i0 and i0 >= 2 and i2 <= M; S1[1, 1, i2] : M >= 3 and i2 <= M and i2 >= 2 }"
child:
  context: "[M] -> { [] : M >= 3 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i0)]; S4[i0, i1] -> [(i0)]; S3[i0, i1] -> [(i0)]; S2[i0, i1, i2] -> [(i0)] }, { S1[i0, i1, i2] -> [(i1)]; S4[i0, i1] -> [(i1)]; S3[i0, i1] -> [(i1)]; S2[i0, i1, i2] -> [(i1)] }, { S1[i0, i1, i2] -> [(i2)]; S4[i0, i1] -> [(0)]; S3[i0, i1] -> [(0)]; S2[i0, i1, i2] -> [(i2)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1, i2] }"
      - filter: "[M] -> { S2[i0, i1, i2] }"
      - filter: "[M] -> { S3[i0, i1] }"
      - filter: "[M] -> { S4[i0, i1] }"
