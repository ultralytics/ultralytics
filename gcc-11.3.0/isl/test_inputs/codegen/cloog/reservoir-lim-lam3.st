domain: "[M] -> { S4[i0] : i0 >= 1 and i0 <= M; S3[i0, i1] : i0 <= M and i1 >= 1 and i1 <= -1 + i0; S2[i0, i1] : i0 <= M and i1 >= 1 and i1 <= -1 + i0; S1[i0, i1, i2] : i0 <= M and i1 <= -1 + i0 and i2 >= 1 and i2 <= -1 + i1 }"
child:
  context: "[M] -> { [] : M >= 1 }"
  child:
    schedule: "[M] -> [{ S4[i0] -> [(5i0)]; S1[i0, i1, i2] -> [(5 + i0 + 2i1 + 2i2)]; S3[i0, i1] -> [(1 + 3i0 + 2i1)]; S2[i0, i1] -> [(3 + i0 + 4i1)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0, i1, i2] }"
        child:
          schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
          child:
            schedule: "[M] -> [{ S1[i0, i1, i2] -> [(i2)] }]"
            options: "[M] -> { separate[i0] }"
      - filter: "[M] -> { S2[i0, i1] }"
        child:
          schedule: "[M] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
      - filter: "[M] -> { S4[i0]; S3[i0, i1] }"
        child:
          schedule: "[M] -> [{ S4[i0] -> [(0)]; S3[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
