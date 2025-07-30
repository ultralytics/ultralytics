domain: "[M] -> { S4[i0, i0] : M >= 3 and i0 <= M and i0 >= 2; S1[i0, 1] : M >= 3 and i0 <= M and i0 >= 2; S3[1, 1] : M >= 3; S2[i0, i1] : i1 <= -1 + i0 and i1 >= 2 and i0 <= M }"
child:
  context: "[M] -> { [] : M >= 3 }"
  child:
    schedule: "[M] -> [{ S1[i0, i1] -> [(i0)]; S4[i0, i1] -> [(i0)]; S3[i0, i1] -> [(i0)]; S2[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)]; S4[i0, i1] -> [(i1)]; S3[i0, i1] -> [(i1)]; S2[i0, i1] -> [(i1)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1] }"
      - filter: "[M] -> { S2[i0, i1] }"
      - filter: "[M] -> { S3[i0, i1] }"
      - filter: "[M] -> { S4[i0, i1] }"
