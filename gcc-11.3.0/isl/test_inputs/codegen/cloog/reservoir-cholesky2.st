domain: "[M] -> { S3[i0, i1, i2] : i0 >= 1 and i1 <= M and i2 >= 1 + i0 and i2 <= i1; S2[i0, i1] : i0 >= 1 and i1 >= 1 + i0 and i1 <= M; S1[i0] : i0 >= 1 and i0 <= M }"
child:
  context: "[M] -> { [] }"
  child:
    schedule: "[M] -> [{ S1[i0] -> [(-1 + 3i0)]; S3[i0, i1, i2] -> [(-1 + i0 + i1 + i2)]; S2[i0, i1] -> [(-2 + 2i0 + i1)] }]"
    options: "[M] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[M] -> { S1[i0] }"
      - filter: "[M] -> { S3[i0, i1, i2] }"
        child:
          schedule: "[M] -> [{ S3[i0, i1, i2] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
          child:
            schedule: "[M] -> [{ S3[i0, i1, i2] -> [(i2)] }]"
            options: "[M] -> { separate[i0] }"
      - filter: "[M] -> { S2[i0, i1] }"
        child:
          schedule: "[M] -> [{ S2[i0, i1] -> [(i1)] }]"
          options: "[M] -> { separate[i0] }"
