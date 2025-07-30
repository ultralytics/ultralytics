domain: "[M] -> { S4[i0, i1] : 2i1 = i0 and M >= 3 and i0 <= 2M and i0 >= 4; S1[i0, 1] : M >= 3 and i0 <= 1 + M and i0 >= 3; S3[2, 1] : M >= 3; S2[i0, i1] : 2i1 <= -1 + i0 and i1 >= 2 and i1 >= -M + i0 }"
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
